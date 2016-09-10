**Note:** There is now a registered package [HiddenMarkovModels.jl](https://github.com/BenConnault/HiddenMarkovModels.jl). This package is no longer maintained (but it still works).

## HiddenMarkovModel.jl
This is an old module for fitting Hidden Markov Models in Julia. Also check out [ToyHMM.jl](https://github.com/ahwillia/ToyHMM.jl), for a simple implementation of a discrete HMM in Julia.

### Installation

The package is not registered (yet), so to download the code use:

```julia
Pkg.clone("https://github.com/ahwillia/HiddenMarkovModel.jl")
```

### Example HMM with Gaussian Emissions

The code below creates two Hidden Markov Models: `hmm` and `hmm_true`. A dataset is generated from `hmm_true` and parameters for `hmm` are fit based on this dataset, and are shown to resemble the ground truth (i.e. the parameters of `hmm_true`).

```julia
using HiddenMarkovModel
using Distributions

# Creates a Gaussian HMM with 2 hidden states (default params)
hmm = HMM(2,Normal()) 

## Create some synthetic training data
μ1,μ2 = -20,15 # mean emission of state 1 and state 2
σ1,σ2 = 3,5    # std of emissions in state 1 and state 2

# A is the transition matrix, B is an Array of emission Distributions
A = [ 0.9 0.1 ; 0.8 0.2 ]
B = (Distribution)[ Normal(μ1,σ1) , Normal(μ2,σ2) ]

# Create the HMM and draw 10 thousand samples from it
hmm_true = HMM(A,B)
s,o = generate(hmm_true,10_000)

# s is a vector of integers specifying the hidden states
# o is a vector of floats specifying the observations

# Use Baum-Welch algorithm to fit the parameters of our first
# HMM object (with default parameters) to the synthetic dataset
ll = fit!(hmm,o)

# ll is the log-likelihood at each iteration.
```

The `fit!` command modifies the hmm parameters to fit the observations, `o`. Different datasets and random initializations will produce different solutions, but typical results are shown below:

```julia
julia> hmm.A
2x2 Array{Float64,2}:
 0.212806   0.787194
 0.0994928  0.900507

julia> hmm.B[1]
Normal(μ=14.871948267720121, σ=5.065092399554946)

julia> hmm.B[2]
Normal(μ=-20.05831728323262, σ=3.007059450241878)
```
