function baum_welch!(hmm::HMM, o::Vector{Int}; max_iter=20, tol=1e-6, scaling=true)
	# Convert input appropriately if user provides a single observation sequence
	return baum_welch!(hmm,(Vector{Int})[o];max_iter=max_iter,tol=tol,scaling=scaling)
end

function baum_welch!(hmm::HMM, sequence_matrix::Matrix{Int}; max_iter=20, tol=1e-6, scaling=true, axis=1)
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

function baum_welch!(hmm::HMM, sequences::Vector{Vector{Int}}; max_iter=20, tol=1e-6, scaling=true)
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