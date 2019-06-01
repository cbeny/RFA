using Flux
using Flux: logitcrossentropy, onecold, data
using Base.Iterators: partition
using LinearAlgebra: tr, I
using Printf: @printf
using Random: randperm
using Metalhead
using Metalhead: trainimgs
using Images: channelview, colorview, RGB
using Statistics: mean
using JLD2: @save, @load
using Serialization

using CuArrays    # comment this out if no gpu available
root_dir = ""
#root_dir = "/var/tmp/"

# precision for the covariance matrices (not the neural net)
const myfloat = Float64

# default regularization factor for matrix inverses
const fudge = eps(myfloat)

# Computes the covariances of features
function cov(F, G) 
	n = size(F,2)
	myfloat.(F/n)*F', myfloat.(G/n)*G', myfloat.(F/n)*G'
end

# Relevance of features given their covariances, with regularization parameter ϵ
function relevance(K,L,A; ϵ=fudge) 
	D = ϵ * Matrix{myfloat}(I, size(K)...)
	return tr(((K .+ D)\A)*((L .+ D)\A'))
end

# for Flux Tracker backpropagation
function relevance(K::TrackedArray, L::TrackedArray, A::TrackedArray; ϵ=fudge)
	Flux.Tracker.track(relevance, K, L, A, ϵ=ϵ)
end

# Relevance for tracked arrays with derivative backpropagation
Flux.Tracker.@grad function relevance(K::TrackedArray, L::TrackedArray, A::TrackedArray; ϵ=0.0)

	K, L, A = cpu.((K, L, A))
	D = ϵ * Matrix{myfloat}(I, size(K)...)
	KA = (K .+ D)\A
	LA = (L .+ D)\A'
	KAL = KA/(L .+ D)

	tr(data(KA)*data(LA)), Δ -> gpu.((-Δ*KA*KAL', -Δ*LA*KAL, 2*Δ*KAL))
end


#  infers Y from features of X
function inferY((K,L,A), G, Y; ϵ=fudge) 
	D = ϵ * Matrix{myfloat}(I, size(K)...)
	(Y*G')*((L .+ D)\A')*inv(K .+ D)/size(G,2)
end 


# RFA loss function, computed from the values of features
function irrel(F,G; ϵ=fudge)
	ker = cov(F, G)
	return size(F,1) - relevance(ker..., ϵ=ϵ) 
end

# Peforms the same as net(dataset), but by batches that (may) fit in the gpu
function ongpu(net, dataset, batch_size=200)
	hcat((cpu(net(gpu(dataset[:,I]))) for I in partition(1:size(dataset,2), batch_size))...)
end

# more efficient than JLD2 for large files (such as the model weights)
function bogosave(filename, var...)
	s = open(filename, "w+")
	serialize(s, var)
	close(s)
end

function bogoload(filename)
	s = open(filename)
	var = deserialize(s)
	close(s)
	return var
end

# the cols of W represents predicted weights for categories
# the cols of Y are one-hot boolean encoding of the truth
# returns an array counting the error rate for best out of 1, 2, 3, ...
function inacurracy(W, Y)
	k, n = size(W)
	err = zeros(k)

	for i=1:n
		per = sortperm(W[:, i], rev=true)
		for j=1:k
			Y[per[j], i] && break
			err[j] += 1
		end
	end
	return err/n
end

fcons(flags::AbstractArray, sep="-") = isempty(flags) ? "" : reduce((f,g)->string(f, sep, g), flags)
fcons(flags::Set, sep="-") = fcons(collect(flags), sep)

# The network is VGG16 but with the last CNN block removed because of the 
# small size of the CIFAR10 images
function vggnet(flags...; k=64) 
	flags = flags ∩ (:bn, :do)
	println("vggnet: ", fcons(flags, " "))

	_convor(p) = Conv((3,3), p, relu, pad=(1,1), stride=(1,1))
	_denser(p) = Dense(p.first, p.second, relu)

	convor(p) = :bn ∈ flags ? Chain(_convor(p), BatchNorm(p.second)) : _convor(p)
	denser(p) = :do ∈ flags ? Chain(_denser(p), Dropout(0.5))       : _denser(p)

	return Chain(x -> reshape(x, 32, 32, 3, :),   
		convor(3 => k), convor(k => k), MaxPool((2, 2)),
		convor(k => 2k), convor(2k => 2k), MaxPool((2, 2)),
		convor(2k => 4k), convor(4k => 4k), convor(4k => 4k), MaxPool((2, 2)),
		convor(4k => 8k), convor(8k => 8k), convor(8k => 8k), MaxPool((2, 2)),
		x -> reshape(x, :, size(x, 4)),
		denser(32k => 4096), denser(4096 => 4096), Dense(4096, 10))
end

# test a model trained with RFA
function test_rfa(dfeatX, dfeatY, X, Y, tX, tY; batch_size = 200)

	# compute the features on the training dataset
	F = ongpu(dfeatX, X, batch_size)
	G = ongpu(dfeatY, Y, batch_size)
	
	# in order to get the inference matrix
	D = inferY(cov(F, G), G, Y)

	# since we have F, we might as well compute the exact training loss
	train_loss = irrel(F, G)
	
	# for testing, we also need the features of the training dataset
	tF = ongpu(dfeatX, tX, batch_size)
	tG = ongpu(dfeatY, tY, batch_size)

	# to compute the test loss...
	test_loss = irrel(tF, tG)
	
	# ... and the predictions errors
	test_errs = inacurracy(D*tF, tY)
	train_err = mean(Flux.onecold(D*F, 1:10) .!= Flux.onecold(Y, 1:10))

	return test_loss, train_loss, test_errs, train_err
end

# format the cifar dataset, extended with horizontal flips
function cifar10(flags...)
	flags = flags ∩ (:norm, :noflip) 
	println("cifar10: ", fcons(flags, " "))

	function getarray(x) 
		y = Float32.(permutedims(channelview(x), (3, 2, 1)))
		reshape(y, :)
	end

	# inverse of getarray (for displayin the image)
	function getimg(x) 
		y = reshape(x,32,32,3)
		colorview(RGB, permutedims(y, (3, 2, 1))) 
	end

	# per channel normalization of the data
	function normalize(x ; ϵ=1e-8, α=α)
		!norm && return x
		μ = mean(x, dims = [1,2,4])
    	σ² = mean((x .- μ) .^ 2, dims = [1,2,4])
    	return reshape((x .- μ) ./ sqrt.(σ² .+ ϵ), 32^2*3, :), (μ, σ²)
	end

	# fast flip operation (horizontal if getarray uses (3,2,1) permutation)
	F = zeros(Float32, 32, 32)
	for i=1:32  F[33-i,i] = 1  end
	flip(x) = reshape(F*reshape(x, 32, :),:)

	# create the basic training set X, Y
	trainset = trainimgs(CIFAR10)
	X = hcat((getarray(x.img) for x in trainset)...)
	if :noflip ∉ flags
		fX = hcat((flip(getarray(x.img)) for x in trainset)...)
	end
	Y = Flux.onehotbatch([x.ground_truth.class for x in trainset], 1:10)

	# compute the mean and variance per channel
	if :norm ∈ flags
		sX = reshape(X, 32^2, 3, :)
		μ = mean(sX, dims = [1,3])
    	σ² = mean((sX .- μ) .^ 2, dims = [1,3])
    	normalize = x -> reshape((reshape(x, 32^2, 3, :) .- μ) ./ sqrt.(σ² .+ 1e-8), 32^2*3, :)
    else
    	normalize = x -> x
	end

	# make the normalized training dataset, extended with horizontal flips
	X[:] = normalize(X)
	if :noflip ∉ flags
		X = hcat(X, normalize(fX))
		Y = hcat(Y, Y)
	end

	# create the normalized testing set
	testset = valimgs(CIFAR10)
	tX = normalize(hcat([getarray(y.img) for y in testset]...))
	tY = Flux.onehotbatch([y.ground_truth.class for y in testset], 1:10) 

	return X, Y, tX, tY
end

# train and test the model
# flags include :rfa for RFA or :ce or crossentropy objective
# :bn adds batchnorm after convo layers, :do adds dropout after FC layers
# :norm normalizes the data per color channel
# :noflip prevents extending the data
function learn(max_epoch, batch_size, label, opt, flags...; load=false, save=false, ϵ0=0.01, k=64)	

	validflags = Set([:rfa, :ce, :bn, :do, :norm, :noflip])
	flags = Set(flags)
	if !(flags ⊆ validflags)
		error("Flags unknown: ", fcons(setdiff(flags, validflags), " "))
	end

	opt_type = string(typeof(opt))
	opt_param =  string(getfield(opt, 1))
	id = string(label, "-", batch_size, "-", fcons(flags), "-", opt_type, "-", opt_param[3:end], "-", k)
	println("Parameters: ", id)

	file_model = string(root_dir, "model-", id, ".raw")
	file_hist = string(root_dir, "hist-", id, ".jld2")
	
	# prepare the dataset
	println("preparing the data...")
	X, Y, tX, tY = cifar10(flags...)
	num_sample = size(X, 2)
	num_batch = div(num_sample, batch_size)
	println("one epoch contains ", num_batch*batch_size, " samples")

	# load or create the neural network
	if load
		println("loading model from file...")
		epoch, cfeatX = bogoload(file_model)
		featX = cfeatX |> gpu
		epoch0 = epoch + 1
		println("starting at epoch ", epoch0)
	else
		println("preparing the model...")
		featX = vggnet(collect(flags)..., k=k)  |> gpu
		epoch0 = 1
	end
	dfeatX = mapleaves(data, featX) 
	Flux.testmode!(dfeatX, true)

	# onehot encoding already forms a complete basis of features
	# so this is just the identity
	dfeatY(x) = Float32.(x) 
	featY(x) = TrackedArray(dfeatY(x))

	# parameters to be optimized
	ps = params(featX)

	# scheduling of the RFA regularization factor
	ϵ(epoch) = max(fudge, ϵ0/2^(epoch-1))

	# training loop
	hist = []
	for epoch in epoch0:max_epoch

		# iterator over random data batches
		parts = partition(randperm(batch_size * num_batch), batch_size)
		dataset = ((X[:,I], Y[:,I]) for I in parts)

		# train for one epoch
		batch = 0
		Flux.testmode!(featX, false)
		Δt = @elapsed Flux.train!(ps, dataset, opt) do x,y
			batch += 1
			print("epoch ", epoch, ": ", batch, "/", num_batch, "\r") 
			if :rfa ∈ flags  
				# RFA loss
				return irrel(featX(gpu(x)), featY(gpu(y)), ϵ = ϵ(epoch))
			else   
				# crossentropy loss
				return logitcrossentropy(featX(gpu(x)), gpu(y))
			end
		end
		Flux.testmode!(featX, true)
		@printf("epoch %d took %.1fs", epoch, Δt)

		# save the model
		if save
			cfeatX = cpu(featX)
			bogosave(file_model, epoch, cfeatX)
		end

		# test the model
		if :rfa ∈ flags   
			# if trained with RFA
			tloss, loss, terrs, err = test_rfa(dfeatX, dfeatY, X, Y, tX, tY)
			push!(hist, (epoch, terrs[1], tloss, err, loss, ϵ(epoch), Δt, terrs))

			@printf("  loss: %.3f/%.3f  inacc: %.1f%%/%.1f%%  inacc(2): %.1f%%  ϵ=%.1e\n", 
				tloss, loss, 100*terrs[1], 100*err, 100*terrs[2], ϵ(epoch))
		else
			# if trained with crossentropy 
			Z = ongpu(dfeatX, tX, batch_size)
			tloss = logitcrossentropy(Z, tY)   
			terrs = inacurracy(Z, tY)

			push!(hist, (epoch, terrs[1], tloss, Δt, terrs))

			@printf("  loss: %.3f  errors: %.1f%%  best of 2 errors: %.1f%%\n", tloss, 100*terrs[1], 100*terrs[2])
		end

		# save the history of test results, and the run's parameters
		@save file_hist hist label flags batch_size ϵ k opt_type opt_param
	end 
end

println("Examples:")
print("RFA loss, no regularization:")
println("\tlearn(25, 100, \"test\", ADAM(), :rfa)")
print("cross-entropy loss, no reg.:")
println("\tlearn(25, 100, \"test\", ADAM(), :ce)")
print("cross-entropy loss, batchnorm:")
println("\tlearn(25, 100, \"test\", ADAM(), :ce, :bn)")