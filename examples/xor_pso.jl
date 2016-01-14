using ArgParse, MPI, Knet, ParticleSwarmOptimization

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

    x = [0.0 0.0 1.0 1.1; 0.0 1.0 0.0 1.1]
    y = [0.0 1.0 1.0 0.0]
    
    @knet function mlp(x; winit=Gaussian(0,.1))
        h1 = wbf(x; out=100, f=:sigm, winit=winit)
        h2 = wbf(h1; out=100, f=:sigm, winit=winit)
        return wbf(h2; out=1, f=:sigm, winit=winit)
    end

    problem = NNProblem(:mlp, args["l"], args["h"], Single(x, y), quadloss)
    
    #last parameter for the parallelization
    gbest = pso(problem, args["num"], args["epoch"], args["phi1"], args["phi2"], args["w"], args["vmin"], args["vmax"], true)

    if rank == 0
        println("\nBest: $(gbest.pbest_fitness)\n")
        println("Gold: $(y)")
        println("Pred: $(forw(gbest.x, x))")
    end

    MPI.Finalize()
end

main() 
