## Test fit with only one hidden state, fit! should equal MLE
hmm = HMM(1,Normal())

# generate some random synthetic data
μ = randn()*10
σ = 1 + rand()*5
o = μ + randn(100)*σ

fit!(hmm, o; tol=NaN)

# use mle, not un-biased estimators
μ_true = mean(o)
σ_true = sqrt(sum((o - mean(o)).^2) / length(o))

@test(round(hmm.B[1].μ,10) == round(μ_true,10))
@test(round(hmm.B[1].σ,10) == round(σ_true,10))

## Test fit on easy synthetic data with two hidden states
μ1,μ2 = -20,15
σ1,σ2 = 3,5

A = [ 0.9 0.1 ;
      0.8 0.2 ]

B = (Distribution)[ Normal(μ1,σ1) , Normal(μ2,σ2) ]

hmm_true = HMM(A,B)
s,o = generate(hmm_true,10_000)

hmm = HMM(2,Normal())
fit!(hmm,o)

@test( (abs(hmm.B[1].μ - μ1) < 0.5 && abs(hmm.B[1].σ - σ1) < 0.5) ||
       (abs(hmm.B[2].μ - μ1) < 0.5 && abs(hmm.B[2].σ - σ1) < 0.5) )
@test( (abs(hmm.B[1].μ - μ2) < 0.5 && abs(hmm.B[1].σ - σ2) < 0.5) ||
       (abs(hmm.B[2].μ - μ2) < 0.5 && abs(hmm.B[2].σ - σ2) < 0.5) )
