module HiddenMarkovModel

using Distributions
using StatsBase:sample,WeightVec
export HMM, generate, forward, backward, viterbi

# HMM constructors
include("HMM.jl")

# Forward/Backward Algorithm and Viterbi estimation of state sequence
include("forward_backward.jl")

# Discrete/toy HMM
# include("dHMM.jl")

end
