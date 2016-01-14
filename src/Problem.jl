using Knet

abstract Data
type Single <: Data
    x
    y
end

type MiniBatch <: Data
    trn
    dev
    tst
    MiniBatch(trn) = new(trn, Any[], Any[])
    MiniBatch(trn, tst) = new(trn, Any[], tst)
end

abstract Problem
type RegularProblem <: Problem
    f::Function
    dim::Int
    l::Float64
    h::Float64
end

type NNProblem <: Problem
    f::Symbol
    l::Float64
    h::Float64
    data::Data
    loss
end
