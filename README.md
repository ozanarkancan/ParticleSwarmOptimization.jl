# ParticleSwarmOptimization

###Introduction

Particle Swarm Optimization (PSO) is a nature inspired heuristic optimization method. In PSO, there are particles, each of them is a solution candidate, search the solution space to find the optimal point of the given function. Each particle has a position and a velocity vector, and searches better position by updating the velocity vector. The velocity updating rule is inspired by bird flocking behaviour and benefits from both particle own best position and the position of the global best particle.

This project contains an implementation of Particle Swarm Optimization in Julia. The repo provides serial and parallel implementation with MPI. The project also provides an interface for training neural networks using PSO with the help of Knet.

###Installation
In Julia repl,

First, install Knet:

Pkg.clone("https://github.com/denizyuret/Knet.jl.git")

Then,

Pkg.clone("https://github.com/ozanarkancan/ParticleSwarmOptimization.jl.git")

Knet has gpu support, and if you want to run your code in a gpu you should build both packages:

Pkg.build("Knet")

Pkg.build("ParticleSwarmOptimization")
