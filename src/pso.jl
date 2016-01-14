using ArgParse, MPI, Knet
include("Particle.jl")

function single_iter(problem::Problem, ps::Array{AbsParticle,1}, gbest::AbsParticle)
    for p in ps
        myfitness = fitness(p, problem)
        if myfitness < p.pbest_fitness
            p.pbest_fitness = myfitness
            copypbest!(p)
                
            if myfitness < gbest.pbest_fitness
                #(MPI.Comm_rank(MPI.COMM_WORLD) == 0) && println("Before: $(forw(gbest.x, problem.x))")
                gbest.pbest_fitness = myfitness
                copygbest!(gbest, p)

                #(MPI.Comm_rank(MPI.COMM_WORLD) == 0) && println("After: $(forw(gbest.x, problem.x))")
            end
        end
    end
end

function search!(problem::Problem, ps::Array{AbsParticle, 1}, gbest::AbsParticle, epoch, phi1, phi2, w, vmin, vmax, parallel)
    if parallel
        comm = MPI.COMM_WORLD
        world = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
    end

    for e=1:epoch
        single_iter(problem, ps, gbest)# calculate local best
        if parallel
            curr_bests = MPI.Allgather([gbest.pbest_fitness], comm)# send each other best fitness value
            #check whether there is a new gbest
            best_p = -1
            for i=1:world
                if curr_bests[i] <= gbest.pbest_fitness
                    gbest.pbest_fitness = curr_bests[i]
                    best_p = i - 1
                end
            end

            #if there is a new gbest broadcast it
            if best_p != -1
                #(MPI.Comm_rank(MPI.COMM_WORLD) == 0) && println("$(best_p) Before: $(forw(gbest.x, problem.x))")
                Bcast!(gbest, best_p, comm)
                #(MPI.Comm_rank(MPI.COMM_WORLD) == 0) && println("$(best_p) After : $(forw(gbest.x, problem.x))")
            end
        end
        
        update!(problem, ps, gbest, phi1, phi2, w, vmin, vmax)
    end
end

function pso(problem::Problem, num = 100, epoch = 100, phi1=2.0, phi2=2.0, w=0.8, vmin=-1.0, vmax=1.0, parallel=false)

    if parallel
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        world = MPI.Comm_size(comm)

        num = Int(floor(num / world))
    end
    
    ps, gbest = init_particles(num, problem)

    if parallel
        (rank == 0) && tic()
    else
        tic()
    end

    search!(problem, ps, gbest, epoch, phi1, phi2, w, vmin, vmax, parallel)
    
    if parallel
        (rank == 0) && toc()
    else
        toc()
    end
    
    return gbest
end
