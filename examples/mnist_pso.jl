using ArgParse, MPI, Knet, ParticleSwarmOptimization
include("mnist.jl")

function minibatch(x, y, batchsize)
    data = Any[]
    for i=1:batchsize:size(x,2)-batchsize+1
        j=i+batchsize-1
        push!(data, (x[:,i:j], y[:,i:j]))
    end
    return data
end

@knet function mnist2layer(x; winit=Gaussian(0,.1))
    h = wbf(x; out=64, f=:relu, winit=winit)
    return wbf(h; out=10, f=:soft, winit=winit)
end

function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    sumloss / numloss
end

function main()
    MPI.Init()
    args = parse_commandline()
    gpu(args["gpu"])
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world = MPI.Comm_size(comm)

    if rank == 0
        println("Parsed args:")
    
        for (arg,val) in args
            println("  $arg  =>  $val")
        end
    end
    
    dtrn = minibatch(identity(MNIST.xtrn), identity(MNIST.ytrn), args["batchsize"])
    dtst = minibatch(identity(MNIST.xtst), identity(MNIST.ytst), args["batchsize"])

    problem = NNProblem(:mnist2layer, args["l"], args["h"], MiniBatch(dtrn), softloss)
    
    #last parameter for the parallelization
    gbest = pso(problem, args["num"], args["epoch"], args["phi1"], args["phi2"], args["w"], args["vmin"], args["vmax"], true)

    if rank == 0
        println("\nBest: $(gbest.pbest_fitness)\n")
        println("Acc: $(1 - test(gbest.x, dtst, zeroone))")
    end

    MPI.Finalize()
end

main() 
