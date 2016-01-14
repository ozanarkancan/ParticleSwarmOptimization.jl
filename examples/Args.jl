using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--dim"
            help = "number of dimensions of the problem"
            default = 100
            arg_type = Int
        
        "--phi1"
            help = "effect of the local best for the velocity update"
            default = 2.0
            arg_type = Float64

        "--phi2"
            help = "effect of the global best for the velocity update"
            default = 2.0
            arg_type = Float64

        "--num"
            help = "number of the particles"
            default = 1024
            arg_type = Int

        "--epoch"
            help = "number of epochs"
            default = 1000
            arg_type = Int

        "--l"
            help = "lower bound of x"
            default = -100.
            arg_type = Float64
        
        "--h"
            help = "upper bound of x"
            default = 100.
            arg_type = Float64

        "--w"
            help = "inertia value"
            default = 0.8
            arg_type = Float64
        
        "--f"
            help = "function that will be optimized"
            default = "qing"

        "--vmin"
            help = "lower bound for the velocity"
            default = -1.0
            arg_type = Float64

        "--vmax"
            help = "upper bound for the velocity"
            default = 1.0
            arg_type = Float64
        
        "--vnorm"
            help = "norm constraint for the velocity"
            default = 1.0
            arg_type = Float64

        "--batchsize"
            help = "batch size"
            arg_type=Int
            default=100
        "--gpu"
            action = :store_true
    end
    return parse_args(s)
end
