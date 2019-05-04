using Flux
using Flux.Tracker: data, TrackedArray, track, @grad
using Flux.Data.MNIST 
using Base.Iterators
using LinearAlgebra 
using Printf 
using Random 
using CuArrays 


# Computes the covariances of features. 
# F and G are of shape (number of features, number of data points)
function cov(F, G) 
	n = size(F,2)
	F/n*F', G/n*G', F/n*G'
end

# Relevance of features given their covariances
relevance(K,L,A) = tr((K\A)*(L\A'))

# Same but with bakprop (doing the matrix inversions on cpu)
relevance(K::TrackedArray, L::TrackedArray, A::TrackedArray) = track(relevance, K, L, A)
@grad function relevance(K::TrackedArray, L::TrackedArray, A::TrackedArray)
	K, L, A = cpu.((K, L, A))
	KA = K\A
	LA = L\A'
	KAL = KA/L
	tr(data(KA)*data(LA)), Δ -> gpu.((-Δ*KA*KAL', -Δ*LA*KAL, 2*Δ*KAL))
end

# produces a matrix which maps a vector of features on Y
# to the infered (expected) value of X
inferX((K,L,A), F, X) = (X*F')*(K\A)*inv(L)/size(F,2)

# conversely, infers Y from features of X
inferY((K,L,A), G, Y) = (Y*G')*(L\A')*inv(K)/size(G,2)

# computes the singular values of the inference maps
function singvals((K,L,A)) 
	isK = inv(sqrt(K))
	M = real.(isK*A*(L\A')*isK)
	return eigvals(Symmetric(M))
end

# Peforms the same as net(dataset), but avoids choking out of memory 
# when using convolutional nets
const test_batch_size = 200 
function apply_by_batch(net, dataset)
	hcat((cpu(net(gpu(dataset[:,I]))) for I in partition(1:size(dataset,2), test_batch_size))...)
end


# do supervised learning on MNIST
function train(max_epoch; cnn=false)

	batch_size = 200 
	opt = ADAM()  

	# let's prepare the data
	float32(x) = Float32.(x)
	images = MNIST.images()
	width, height = size(images[1]) # = 28, 28

	# training dataset
	X = reshape(hcat(float32.(images)...), width*height, :)
	Y = Flux.onehotbatch(MNIST.labels(), 0:9) 

	# test dataset
	tX = reshape(hcat(float32.(MNIST.images(:test))...), width*height, :)
	tY = Flux.onehotbatch(MNIST.labels(:test), 0:9) 

	# for MNIST: num_feat=10, num_data=60,000 
	num_feat, num_data = size(Y)
	mum_batch = div(num_data, batch_size)

	# neural net that extract features on images
	if cnn
		println("Using a convolutional neural net.")

		featX = Chain(x->reshape(x, width, height, 1, :),
				  	Conv((3,3), 1=>32, relu),
				  	Conv((4,4), 32=>64, stride=(2,2), relu),
				  	Conv((3,3), 64=>64, relu),
				  	Conv((4,4), 64=>128, stride=(2,2), relu),
				  	x->reshape(x, :, size(x,4)),
				  	Dense(2048, 1024, relu), 
				  	Dense(1024, num_feat))  |> gpu
	else
		println("Using a 2-layer perceptron.")

		featX = Chain(Dense(width*height, 1024, relu), 
				  	Dense(1024, 1024, relu),
				  	Dense(1024, num_feat)) |> gpu
	end

	# The label's one hot encoding already serve as optimal features
	# We just need them to be "tracked" to avoid a Flux bug
	# (ambiguity in overloading of matrix multiplication)
	featY(x) = TrackedArray(Float32.(x)) |> gpu

	# parameters to be optimized, in general one would include featY as well
	ps = params(featX)	


	# loss function	
	function loss(x,y) 
		ker = cov(featX(x), featY(y))
		return num_feat - relevance(ker...) 
	end

	# We'll use them outside the loop
	local test_kernel, I

	# learning loop
	for epoch in 1:max_epoch

		print("epoch ", epoch, "/", max_epoch, "... ")

		# iterator over random data batches
		parts = partition(randperm(num_data), batch_size)
		dataset = ((gpu(X[:,I]), gpu(Y[:,I])) for I in parts)

		# train for one epoch
		Flux.testmode!(featX, false)
		Δt = @elapsed Flux.train!(loss, ps, dataset, opt)
   
		Flux.testmode!(featX, true)

		print("testing...\r")

		# let's compute the features of the test data 
		tF = apply_by_batch(data∘featX, tX) 
		tG = apply_by_batch(data∘featY, tY) 
		test_kernel = cov(tF, tG)

		# compute the test loss
		test_loss = num_feat - relevance(test_kernel...) 

		@printf("epoch %d took %.2fs  test loss = %.3f  ", epoch, Δt, test_loss)

		# build the inference matrix I
		F = apply_by_batch(data∘featX, X)
		G = apply_by_batch(data∘featY, Y)
		kernel = cov(F, G)
		I = inferY(kernel, G, Y)

		# infer labels and compare with truth
		test_err = sum(Flux.onecold(I*tF) .!= Flux.onecold(tY))

		@printf("test errors = %.2f%%\n",  
			100*test_err/size(tX,2))
	end 

	# let's return the prediction model, and final relevances
	return x -> Flux.onecold(I*data(featX(x)), 0:9), singvals(test_kernel)
end

model, λ = train(6, cnn=true)
