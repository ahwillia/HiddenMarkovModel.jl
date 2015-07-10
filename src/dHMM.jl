# Discrete HMM -- HMM with discrete emission probabilities
type dHMM
	n::Int             # Number of hidden states
	m::Int 			   # Alphabet length (observations are discrete for now)
	A::Matrix{Float64} # Estimated state-transition matrix A[i,j] = Pr[i->j]
	B::Matrix{Float64} # Estimated emission probabilities (discrete for now)
	p::Vector{Float64} # Estimiated initial state probabilities
	
	# Notes:
	#   "A" is a NxN matrix, rows sum to one
	#   "B" is a NxM matrix, rows sum to one

	# To do:
	#    Allow B to depend on other observables, for observation o and param c, B(o|c)
end

function dHMM(n::Int,m::Int)
	# Randomize state-transition matrix
	A = rand(n,n)
	A ./= repmat(sum(A,1),n,1) # normalize rows
	
	# Randomize emission probability matrix
	B = rand(n,m)
	B ./= repmat(sum(B,2),1,m) # normalize rows
	
	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return dHMM(n,m,A,B,p)
end

function dHMM(A::Matrix{Float64},B::Matrix{Float64})
	# Check dimensions of matrices
	n,m = size(B)
	assert(size(A,1) == size(A,2) == n)

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return dHMM(n,m,A,B,p)
end

function dHMM(A::Matrix{Float64},B::Matrix{Float64},p::Vector{Float64})
	assert(sum(p) == 1)
	hmm = dHMM(A,B)
	hmm.p = p
	return hmm
end

function generate(hmm::dHMM, n_obs::Int)
	# Generate a sequence of n_obs observations from an HMM.

	# Sequence of states and observations
	s = zeros(Int,n_obs) # states
	o = zeros(Int,n_obs) # observations

	# Choose initial state with probabilities weighted by "init_state"
	s[1] = sample(WeightVec(hmm.p))         # hmm.p are the initial state probabilities
	o[1] = sample(WeightVec(vec(hmm.B[s[1],:]))) # draw obs given initial state

	# Iterate drawing observations and updating state
	for i = 2:n_obs
		s[i] = sample(WeightVec(vec(hmm.A[s[i-1],:]))) # Pr(s[i+1]==j|s[i]) = A[j,s[i]]
		o[i] = sample(WeightVec(vec(hmm.B[s[i],:])))   # Pr(o==k|s) = B[s,k]
	end

	# return sequence of states and observations
	return (s,o)
end

function forward(hmm::dHMM, o::Vector{Int}; scaling=true)
	n_obs = length(o)

	# alpha[t,i] = probability of being in state 'i' given o[1:t]
	alpha = zeros(n_obs, hmm.n) 

	# base case (initialize at start)
	for i = 1:hmm.n
		alpha[1,i] = hmm.p[i] * hmm.B[i,o[1]]
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
			alpha[t,j] *= hmm.B[j,o[t]]
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

function backward(hmm::dHMM, o::Vector{Int}; scale_coeff=None)
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
				beta[t,i] += hmm.A[i,j] * hmm.B[j,o[t+1]] * beta[t+1,j]
			end
		end
		if scale_coeff != None
			beta[t,:] *= scale_coeff[t]
		end
	end

	return beta
end

function viterbi(hmm::dHMM, o::Vector{Int})
	n_obs = length(o)

	# delta[i,j] = highest probability of state sequence ending in state j on step i
	# psi[i,j] = most likely state on step i-1 given state j on step i (argmax of deltas)
	delta = zeros(n_obs, hmm.n)
	psi = ones(Int, n_obs, hmm.n)

	# base case, psi[:,1] is ignored so don't initialize
	for i = 1:hmm.n
		delta[1,i] = hmm.p[i] .* hmm.B[i,o[1]]
	end

	# induction step
	for t = 2:n_obs
		for j = 1:hmm.n
			delta[t,j],psi[t,j] = findmax(hmm.A[:,j].*delta[t-1,:]')
			delta[t,j] *= hmm.B[j,o[t]]
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
	
function baum_welch!(hmm::dHMM, o::Vector{Int}; max_iter=20, tol=1e-6, scaling=true)
	# Convert input appropriately if user provides a single observation sequence
	return baum_welch!(hmm,(Vector{Int})[o];max_iter=max_iter,tol=tol,scaling=scaling)
end

function baum_welch!(hmm::dHMM, sequence_matrix::Matrix{Int}; max_iter=20, tol=1e-6, scaling=true, axis=1)
	# Convert input appropriately if user provides a matrix of observations sequences
	sequences = (Vector{Int})[]

	# Let the user specify whether sequences are columns or row (default is columns)
	if axis == 1
		for i = 1:size(sequence_matrix,2)
			push!(sequences,sequence_matrix[:,i])
		end
	elseif axis == 2
		for i = 1:size(sequence_matrix,1)
			push!(sequences,sequence_matrix[i,:])
		end
	else
		error("axis argument not valid. Must be 1, to specify sequences as columns, or 2, to specify sequences as rows")
	end

	# Fit the hmm now that sequences converted to Vector{Vector{Int}}
	return baum_welch!(hmm, sequences; max_iter=max_iter, tol=tol, scaling=scaling)
end

function baum_welch!(hmm::dHMM, sequences::Vector{Vector{Int}}; max_iter=20, tol=1e-6, scaling=true)
	# Fit hmm parameters given set of observation sequences
	# Setting tol to NaN will prevent early stopping, resulting in 'max_iter' iterations
	n_seq = length(sequences)

    # convergence history of the fit, log-liklihood
    ch = (Float64)[]

    for k = 1:max_iter
    	push!(ch,0.0)

    	# Store weighted sums for numerators/denominators across sequences
    	An_sum,Bn_sum = zeros(hmm.n,hmm.n),zeros(hmm.n,hmm.m)
    	Ad_sum,Bd_sum = zeros(hmm.n),zeros(hmm.n)
    	p_sum = zeros(hmm.n)

    	for o in sequences
    		# E step
    		log_p_obs,alpha,beta,x,g = calc_stats(hmm,o,scaling)
    		ch[end] += log_p_obs #(log_p_obs + log(length(o)))

    		# M step (re-estimation)
    		An,Ad,Bn,Bd,p_new = re_estimate(hmm,o,x,g)

    		# Add estimates to weighted sums
    		w_k = -1.0 / log_p_obs #(log_p_obs + log(length(o)))
    		An_sum += w_k * An
    		Ad_sum += w_k * Ad
    		Bn_sum += w_k * Bn
    		Bd_sum += w_k * Bd
    		p_sum += w_k * p_new
		end

		# Update parameters, combining across sequences
		for i = 1:hmm.n
			for j = 1:hmm.n
				hmm.A[i,j] = An_sum[i,j] / Ad_sum[i]
			end
			for z = 1:hmm.m
				hmm.B[i,z] = Bn_sum[i,z] / Bd_sum[i]
			end
		end
		hmm.p = p_sum ./ sum(p_sum)
		
		if length(ch)>1 && (ch[end]-ch[end-1] < tol)
			println("Baum-Welch converged, stopping early")
			break
		end
	end

	return ch
end

function re_estimate(hmm,o,x,g)
	# Estimate numerator (An) and denominator (Ad) terms for updating hmm.A 
	An = zeros(hmm.n, hmm.n) # An[i,j] = expected # of transitions from state 'i' to state 'j'
	Ad = zeros(hmm.n)        # Ad[i] = expected # of transitions from state 'i'
	for i = 1:hmm.n
		Ad[i] = sum(g[1:end-1,i]) 
		for j = 1:hmm.n
			An[i,j] = sum(x[:,i,j])
		end
	end

	# Estimate numerator (Bn) and denominator (Bd) terms for updating hmm.B
	Bn = zeros(hmm.n, hmm.m) # Bn[i,z] = expected # of times observing 'z' in state 'i'
	Bd = sum(g,1)'           # Bd[i] = expected # of time steps in state 'i'
	for i = 1:hmm.n
		for z = 1:hmm.m
			Bn[i,z] += sum(g[o.==z,i])
		end
	end

	# Re-estimate hmm.p (initial state probabilities)
	p_new = vec(g[1,:])

	return (An,Ad,Bn,Bd,p_new)
end

function calc_stats(hmm,o::Vector{Int},scaling)
	## Single E-M iteration in the Baum-Welch procedure
	n_obs = length(o)

	# Calculate forward/backward probabilities
	if scaling
		alpha, log_p_obs, coeff = forward(hmm,o; scaling=true)
		beta = backward(hmm,o; scale_coeff=coeff)
	else
		alpha, p_obs = forward(hmm,o; scaling=false)
		log_p_obs = log(p_obs)
	    beta = backward(hmm,o)
	end

	# x[t,i,j] = probability of being in state 'i' at 't' and then in state 'j' at 't+1'
	x = zeros(n_obs-1, hmm.n, hmm.n)
	for t = 1:(n_obs-1)
		for i = 1:hmm.n
			for j = 1:hmm.n
				x[t,i,j] = alpha[t,i] * hmm.A[i,j] * hmm.B[j,o[t+1]] * beta[t+1,j]
			end
		end
		x[t,:,:] ./= sum(x[t,:,:]) # normalize to achieve probabilities
	end

	# g[t,i] = probability of being in state 'i' at step 't' given all observations
	g = alpha .* beta
	g ./= sum(g,2)   # normalize across states

	return log_p_obs, alpha, beta, x, g
end