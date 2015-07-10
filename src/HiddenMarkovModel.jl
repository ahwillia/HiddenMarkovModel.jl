module HiddenMarkovModel

using Distributions
using StatsBase:sample,WeightVec
export HMM, generate, forward, backward, viterbi

# Flexible HMM implementation
include("HMM.jl")

# Discrete/toy HMM
# include("dHMM.jl")

end
