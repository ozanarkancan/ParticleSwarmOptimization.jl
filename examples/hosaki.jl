using Knet, ParticleSwarmOptimization
include("Args.jl")

hosaki(x) = (1 - 8*x[1] + 7*(x[1]^2) - 7/3*(x[1]^3) + (1/4)*(x[1]^4))*(x[2]^2)*exp(-1 * x[2])
#x* = (4,2)
#f(x*) = -2.3458
#julia hosaki.jl --num 10 --epoch 100

function main()
    gpu(false)
    args = parse_commandline()
    
    problem = RegularProblem(hosaki, 2, [0.0,0.0], [5.0,6.0])
    gbest = pso(problem, args["num"], args["epoch"], args["phi1"], args["phi2"], args["w"], args["vmin"], args["vmax"])
    
    println("\nBest: $(gbest.pbest_fitness), $(gbest.x)")
end

main()

