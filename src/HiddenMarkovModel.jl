module HiddenMarkovModel

using Distributions
using StatsBase:sample,WeightVec

# HMM constructors
export HMM, generate
include("HMM.jl")

# Forward/Backward Algorithm and Viterbi estimation of state sequence
export forward_backward, viterbi
include("forward_backward.jl")

# Fitting algorithms (under development)
export fit!
include("fit.jl")

# Discrete/toy HMM -- include("dHMM.jl")

end
