module ParticleSwarmOptimization

include("pso.jl"); export pso, single_iter, search!
include("Particle.jl"); export AbsParticle, Particle, NNParticle, fitness, init_particles, update!, Bcast!, copybest!, copygbest!
include("Problem.jl"); export Data, Single, MiniBatch, Problem, RegularProblem, NNProblem

end # module
