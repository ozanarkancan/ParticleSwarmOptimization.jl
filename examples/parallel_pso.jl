using ArgParse, MPI, Knet, ParticleSwarmOptimization

function main()
    MPI.Init()
    gpu(false)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world = MPI.Comm_size(comm)
    args = parse_commandline()

    if rank == 0
        println("Parsed args:")
    
        for (arg,val) in args
            println("  $arg  =>  $val")
        end
    end

    problem = RegularProblem(eval(parse(args["f"])), args["dim"], args["l"], args["h"])
    
    #last parameter for the parallelization
    gbest = pso(problem, args["num"], args["epoch"], args["phi1"], args["phi2"], args["w"], args["vmin"], args["vmax"], true)

    rank == 0 && println("\nBest: $(gbest.pbest_fitness)\n$(gbest.x)")

    MPI.Finalize()
end

main() 
