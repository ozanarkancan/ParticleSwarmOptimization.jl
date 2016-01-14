using ArgParse, Knet, ParticleSwarmOptimization
include("Args.jl")
include("Functions.jl")

function main()
    gpu(false)
    args = parse_commandline()
    println("Parsed args:")
    
    for (arg,val) in args
        println("  $arg  =>  $val")
    end

    problem = RegularProblem(eval(parse(args["f"])), args["dim"], args["l"], args["h"])
    gbest = pso(problem, args["num"], args["epoch"], args["phi1"], args["phi2"], args["w"], args["vmin"], args["vmax"])
    
    println("\nBest: $(gbest.pbest_fitness), $(gbest.x)")
end

main()

