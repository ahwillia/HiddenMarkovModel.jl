module HiddenMarkovModel

using Distributions
using StatsBase:sample,WeightVec
export HMM, generate, forward, backward, viterbi, fit!

# HMM constructors
include("HMM.jl")

# Forward/Backward Algorithm and Viterbi estimation of state sequence
include("forward_backward.jl")

# Fitting algorithms (under development)
include("fit.jl")

# Discrete/toy HMM
# include("dHMM.jl")

end
