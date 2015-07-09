module HiddenMarkovModel

using StatsBase:sample,WeightVec
export dHMM, generate, forward, backward, viterbi, baum_welch!

# Discrete/toy HMM
include("dHMM.jl")

end
