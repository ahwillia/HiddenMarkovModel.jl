# HMM -- HMM with emission probability distribution 'C'
type HMM{C<:Distribution}
	n::Int             # Number of hidden states
	A::Matrix{Float64} # Estimated state-transition matrix A[i,j] = Pr[i->j]
	B::Vector{C}       # Estimated emission probability distributions
	p::Vector{Float64} # Estimiated initial state probabilities
	
	# Notes:
	#   "A" is a NxN matrix, rows sum to one
	#   "B" is a NxM matrix, rows sum to one

	# To do:
	#    Allow B to depend on other observables, for observation o and param c, B(o|c)
end

function HMM(n::Int,C::Distribution)
	# Randomize state-transition matrix
	A = rand(n,n)
	A ./= repmat(sum(A,1),n,1) # normalize rows	
	
	# Specify a distribution of type C for each state
	B = fill(C,n)

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return HMM(n,A,B,p)
end

function HMM(A::Matrix{Float64},C::Distribution)
	# determine number of states
	@assert size(A,1) == size(A,2)
	n = size(A,1)

	# Specify a distribution of type C for each state
	B = fill(C,n)

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return HMM(n,A,B,p)
end

function HMM(A::Matrix{Float64},Bmat::Matrix{Float64})
	# Initialize a discrete HMM, with Categorical() emissions
	n = size(A,1)

	# Each row of Bmat specifies a Categorical pdf
	B = (Distribution)[]
	for i = 1:n
		push!(B,Categorical(vec(Bmat[i,:])))
	end
	
	return HMM(A,B)
end

function HMM(A::Matrix{Float64},B::Vector{Distribution})
	# determine number of states
	@assert size(A,1) == size(A,2)
	n = size(A,1)
	@assert length(B) == n

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return HMM(n,A,B,p)
end

function HMM(A::Matrix{Float64},B,p::Vector{Float64})
	@assert sum(p)==1
	hmm = HMM(A,B)
	hmm.p = p # reset initial probabilities
	return hmm
end

function generate(hmm::HMM, n_obs::Int)
	# Generate a sequence of n_obs observations from an HMM.

	# Sequence of states and observations
	s = zeros(Int,n_obs) # states
	o = zeros(n_obs)     # observations

	# Choose initial state with probabilities weighted by "init_state"
	s[1] = rand(Categorical(hmm.p))  # hmm.p are the initial state probabilities
	o[1] = rand(hmm.B[s[1]])         # draw observation given initial state

	# Construct categorical distributions from each row of A
	Ac = (Categorical)[]
	for i = 1:hmm.n
		push!(Ac,Categorical(vec(hmm.A[i,:])))
	end

	# Iterate drawing observations and updating state
	for t = 2:n_obs
		s[t] = rand(Ac[s[t-1]])   # sample from state-transition matrix
		o[t] = rand(hmm.B[s[t]])  # sample from emission probability distribution
	end

	# return sequence of states and observations
	return (s,o)
end

function forward(hmm::HMM, o::Vector; scaling=true)
	n_obs = length(o)

	# alpha[t,i] = probability of being in state 'i' given o[1:t]
	alpha = zeros(n_obs, hmm.n) 

	# base case (initialize at start)
	for i = 1:hmm.n
		alpha[1,i] = hmm.p[i] * pdf(hmm.B[i],o[1])
	end

	if scaling
		c = (Float64)[] # scaling coefficients
		push!(c,1./sum(alpha[1,:]))
		alpha[1,:] *= c[end] 
	end

	# induction step
	for t = 2:n_obs
		for j = 1:hmm.n
			for i = 1:hmm.n
				alpha[t,j] += hmm.A[i,j] * alpha[t-1,i]
			end
			alpha[t,j] *= pdf(hmm.B[j],o[t])
		end
		if scaling
			push!(c,1./sum(alpha[t,:]))
			alpha[t,:] *= c[end]
		end
	end

	# Calculate likelihood (or log-likelihood) of observed sequence
	if scaling
		log_p_obs = -sum(log(c)) # see Rabiner (1989), eqn 103
		return (alpha,log_p_obs,c)
	else
		p_obs = sum(alpha[end,:]) 
		return (alpha,p_obs)
	end
end

function backward(hmm::HMM, o::Vector; scale_coeff=None)
	# scale_coeff are 1/sum(alpha[t,:]) calculated by forward algorithm
	n_obs = length(o)

	# beta[t,i] = probability of being in state 'i' and then obseverving o[t+1:end]
	beta = zeros(n_obs, hmm.n)

	# base case (initialize at end)
	if scale_coeff == None
		beta[end,:] += 1
	else
		if length(scale_coeff) != n_obs
			error("scale_coeff is improperly defined (wrong length)")
		end
		beta[end,:] += scale_coeff[end]
	end

	# induction step
	for t = reverse(1:n_obs-1)
		for i = 1:hmm.n
			for j = 1:hmm.n
				beta[t,i] += hmm.A[i,j] * pdf(hmm.B[j],o[t+1]) * beta[t+1,j]
			end
		end
		if scale_coeff != None
			beta[t,:] *= scale_coeff[t]
		end
	end

	return beta
end

function viterbi(hmm::HMM, o::Vector)
	n_obs = length(o)

	# delta[i,j] = highest probability of state sequence ending in state j on step i
	# psi[i,j] = most likely state on step i-1 given state j on step i (argmax of deltas)
	delta = zeros(n_obs, hmm.n)
	psi = ones(Int, n_obs, hmm.n)

	# base case, psi[:,1] is ignored so don't initialize
	for i = 1:hmm.n
		delta[1,i] = hmm.p[i] .* pdf(hmm.B[i],o[1])
	end

	# induction step
	for t = 2:n_obs
		for j = 1:hmm.n
			delta[t,j],psi[t,j] = findmax(hmm.A[:,j].*delta[t-1,:]')
			delta[t,j] *= pdf(hmm.B[j],o[t])
		end
	end

	# backtrack to uncover the most likely path / state sequence
	q = zeros(Int,n_obs) # vector holding state sequence
	q[end] = indmax(delta[end,:])

	# backtrack recursively
	for t = reverse(1:n_obs-1)
		q[t] = psi[t+1,q[t+1]]
	end
	return q
end
