using Base.LinAlg: axpy!, scale!
using Knet, MPI 
@useifgpu CUDArt
include("Problem.jl")

abstract AbsParticle

type Particle <: AbsParticle
    x::Array{Float64,1}
    v::Array{Float64,1}
    pbest::Array{Float64, 1}
    pbest_fitness::Float64
   
    Particle(x::Array{Float64, 1}) = new(x, similar(x), similar(x), 1e15)
    Particle(x::Array{Float64, 1}, v::Array{Float64, 1}) = new(x, v, similar(x), 1e12)
end

type NNParticle <: AbsParticle
    x::Knet.Net
    v::Knet.Net
    pbest::Knet.Net
    pbest_fitness::Float64
    
    NNParticle(x::Knet.Net) = new(x, Knet.Net(similar(x.reg)), Knet.Net(similar(x.reg)), 1e15)
    NNParticle(x::Knet.Net, v::Knet.Net) = new(x, v, deepcopy(x), 1e12)
    NNParticle(x::Knet.Net, v::Knet.Net, pbest::Knet.Net) = new(x, v, pbest, 1e12)
end

fitness(p::Particle, problem::RegularProblem) = problem.f(p.x)
fitness(p::NNParticle, problem::NNProblem) = fitness(p, problem, problem.data)
fitness(p::NNParticle, problem::NNProblem, data::Single) = problem.loss(forw(p.x, problem.data.x), problem.data.y)

function fitness(p::NNParticle, problem::NNProblem, data::MiniBatch)
    sumloss = numloss = 0
    for (x,ygold) in data.trn
        ypred = forw(p.x, x)
        sumloss += problem.loss(ypred, ygold)
        numloss += 1
    end
    return sumloss / numloss
end

function uniform(dim, l, h)
    arr = rand(dim)
    arr = arr * (h - l) + l
end

function uniform(dim, l::Array{Float64, 1}, h::Array{Float64, 1})
    arr = rand(dim)
    low = maximum(l)
    high = minimum(h)
    arr = arr * (high - low) + l
end


function init_particles(num, problem::RegularProblem)
    ps = AbsParticle[]
    for p=1:num
        push!(ps, Particle(uniform(problem.dim, problem.l, problem.h), uniform(problem.dim, -1.0 , 1.0)))
    end
    gbest = Particle(uniform(problem.dim, problem.l, problem.h))
    return ps, gbest
end

init_particles(num, problem::NNProblem) = init_particles(num, problem, problem.data)

function init_particles(num, problem::NNProblem, Data::Single)
    ps = AbsParticle[]
    for p=1:num
        net1 = compile(problem.f);
        forw(net1, problem.data.x[:, 1])
        net2 = compile(problem.f);
        forw(net2, problem.data.x[:, 1])
        net3 = compile(problem.f);
        forw(net3, problem.data.x[:, 1])
        push!(ps, NNParticle(net1, net2, net3))
    end
    
    net = compile(problem.f);
    forw(net, problem.data.x[:,1])

    gbest = NNParticle(net)
    return ps, gbest
end

function init_particles(num, problem::NNProblem, Data::MiniBatch)
    ps = AbsParticle[]
    for p=1:num
        net1 = compile(problem.f);
        forw(net1, problem.data.trn[1][1])
        net2 = compile(problem.f);
        forw(net2, problem.data.trn[1][1])
        net3 = compile(problem.f);
        forw(net3, problem.data.trn[1][1])
        push!(ps, NNParticle(net1, net2, net3))
    end
    
    net = compile(problem.f);
    forw(net, problem.data.trn[1][1])
    gbest = NNParticle(net)
    return ps, gbest
end

function update!(px, pv, pbest, gbest, w, l::Float64, h::Float64, vmin, vmax, r1, r2)
    scale!(pv, w)
    axpy!(1, r1 * (pbest - px), pv)
    axpy!(1, r2 * (gbest - px), pv)
    for i=1:length(pv); pv[i] = pv[i] < vmin ? vmin : pv[i] > vmax ? vmax : pv[i] end
    axpy!(1, pv, px)
    for i=1:length(px); px[i] = px[i] < l ? l : px[i] > h ? h : px[i] end
end

function update!(px, pv, pbest, gbest, w, l::Array{Float64, 1}, h::Array{Float64, 1}, vmin, vmax, r1, r2)
    scale!(pv, w)
    axpy!(1, r1 * (pbest - px), pv)
    axpy!(1, r2 * (gbest - px), pv)
    for i=1:length(pv); pv[i] = pv[i] < vmin ? vmin : pv[i] > vmax ? vmax : pv[i] end
    axpy!(1, pv, px)
    for i=1:length(px); px[i] = px[i] < l[i] ? l[i] : px[i] > h[i] ? h[i] : px[i] end
end


const libpso = Libdl.find_library(["libpso"], [Pkg.dir("ParticleSwarmOptimization/src")])

@gpu update!(px::CudaArray{Float32}, pv::CudaArray{Float32}, pbest::CudaArray{Float32}, gbest::CudaArray{Float32}, w, l, h, vmin, vmax, r1, r2)=ccall((:update32,libpso),Void,(Cint,Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),length(px),px, pv, pbest, gbest, w, l, h, vmin, vmax, r1, r2); gpusync();


@gpu update!(px::CudaArray{Float64}, pv::CudaArray{Float64}, pbest::CudaArray{Float64}, gbest::CudaArray{Float64}, w, l, h, vmin, vmax, r1, r2)=ccall((:update64,libpso),Void,(Cint,Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),length(px),px, pv, pbest, gbest, w, l, h, vmin, vmax, r1, r2); gpusync();

function update!(problem::RegularProblem, p::Particle, gbest::Particle, phi1, phi2, w, vmin, vmax)
    l = problem.l
    h = problem.h
    update!(p.x, p.v, p.pbest, gbest.x, w, l, h, vmin, vmax, rand() * phi1, rand() * phi2)
end

function update!(problem::NNProblem, p::NNParticle, gbest::NNParticle, phi1, phi2, w, vmin, vmax)
    l = problem.l
    h = problem.h
    for i=1:length(p.v.reg)
        if isa(p.v.reg[i].op, Knet.Par)
            update!(p.x.reg[i].out, p.v.reg[i].out, p.pbest.reg[i].out, gbest.x.reg[i].out, w, l, h, vmin, vmax, rand() * phi1, rand() * phi2)
        end
    end
end

function update!(problem::Problem, ps::Array{AbsParticle, 1}, gbest::AbsParticle, phi1, phi2, w, vmin, vmax)
    for p in ps
        update!(problem, p, gbest, phi1, phi2, w, vmin, vmax)
    end
end

Bcast!(gbest::Particle, root::Int, comm::MPI.Comm) = MPI.Bcast!(gbest.x, length(gbest.x), root, comm)

addbuffer(buffer, arr) = (length(buffer) == 0) ? vec(arr) : vcat(buffer, vec(arr))
@gpu addbuffer(buffer, arr::CudaArray) = (length(buffer) == 0) ? vec(to_host(arr)) : vcat(buffer, vec(to_host(arr)))

function Bcast!(gbest::NNParticle, root::Int, comm::MPI.Comm)
    buffer = Any[]
    for i=1:length(gbest.x.reg)
        if isa(gbest.x.reg[i].op, Knet.Par)
            buffer = addbuffer(buffer, gbest.x.reg[i].out)
        end
    end
    #(MPI.Comm_rank(MPI.COMM_WORLD) == 0) && println("Root: $root Buffer1: $(buffer[1:5])")
    MPI.Bcast!(buffer, length(buffer), root, comm)
    #(MPI.Comm_rank(MPI.COMM_WORLD) == 0) && println("Root: $root Buffer2: $(buffer[1:5])")
    start = 1
    for i=1:length(gbest.x.reg)
        if isa(gbest.x.reg[i].op, Knet.Par)
            rs = size(gbest.x.reg[i].out)
            l = prod(rs)
            copy!(gbest.x.reg[i].out, reshape(buffer[start:(start+l-1)], rs))
            start += l
        end
    end
end

copypbest!(p::Particle) = copy!(p.pbest, p.x)

function copypbest!(p::NNParticle)
    for i=1:length(p.x.reg)
        if isa(p.x.reg[i].op, Knet.Par)
            copy!(p.pbest.reg[i].out, p.x.reg[i].out)
        end
    end
end

copygbest!(gbest::Particle, p::Particle) = copy!(gbest.x, p.x)

function copygbest!(gbest::NNParticle, p::NNParticle)
    for i=1:length(p.x.reg)
        if isa(p.x.reg[i].op, Knet.Par)
            copy!(gbest.x.reg[i].out, p.x.reg[i].out)
        end
    end
end
