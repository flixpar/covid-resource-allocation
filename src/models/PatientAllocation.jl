module PatientAllocation

using JuMP
using Gurobi

using LinearAlgebra
using MathOptInterface
using Distributions

export patient_allocation, patient_block_allocation


##############################################
############# Standard Model #################
##############################################

function patient_allocation(
		beds::Array{<:Real,1},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		adj_matrix::BitArray{2};
		los=11,
		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		smoothness_penalty::Real=0,
		setup_cost::Real=0,
		sent_penalty::Real=0,
		balancing_thresh::Real=1.0,
		balancing_penalty::Real=0,
		severity_weighting::Bool=false,
		no_artificial_overflow::Bool=false,
		capacity_cushion::Real=-1,
		verbose::Bool=false,
)
	N, T = size(admitted_patients)
	@assert(size(initial_patients, 1) == N)
	@assert(size(beds, 1) == N)
	@assert(size(adj_matrix) == (N,N))
	@assert(size(discharged_patients) == (N, T))

	###############
	#### Setup ####
	###############

	L = nothing
	if isa(los, Int)
		L = vcat(ones(Int, los), zeros(Int, T-los))
	elseif isa(los, Array{<:Real,1})
		if length(los) >= T
			L = los
		else
			L = vcat(los, zeros(Float64, T-length(los)))
		end
	elseif isa(los, Distribution)
		L = 1.0 .- cdf.(los, 0:T)
	else
		error("Invalid length of stay distribution")
	end

	if 0 < capacity_cushion < 1
		beds = beds .* (1.0 - capacity_cushion)
	end

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	###############
	## Variables ##
	###############

	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, obj_dummy[1:N,1:T] >= 0)

	#################
	## Expressions ##
	#################

	# expressions for the number of active patients
	@expression(model, active_patients[i=1:N,t=1:T],
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(L[t-t₁+1] * (
			admitted_patients[i,t₁]
			- sum(sent[i,:,t₁])
			+ sum(sent[:,i,t₁])
		) for t₁ in 1:t)
		+ sum(sent[i,:,t])
	)
	active_null = [(
			initial_patients[i]
			- sum(discharged_patients[i,1:t])
			+ sum(L[t-t₁+1] * admitted_patients[i,t₁] for t₁ in 1:t)
		) for i in 1:N, t in 1:T
	]

	# expression for the patient overflow
	@expression(model, overflow[i=1:N,t=1:T], active_patients[i,t] - beds[i])

	# objective function
	objective = @expression(model, sum(obj_dummy))

	######################
	## Hard Constraints ##
	######################

	# ensure the number of active patients is non-negative
	@constraint(model, [i=1:N,t=1:T], active_patients[i,t] >= 0)

	# only send new patients
	@constraint(model, [t=1:T], sum(sent[:,:,t], dims=2) .<= admitted_patients[:,t])

	# only send patients between connected locations
	for i in 1:N, j in 1:N
		if ~adj_matrix[i,j]
			for t in 1:T
				fix(sent[i,j,t], 0, force=true)
			end
		end
	end

	# objective constraint
	@constraint(model, [i=1:N,t=1:T], obj_dummy[i,t] >= overflow[i,t])

	##########################
	## Optional Constraints ##
	##########################

	# enforce minimum transfer amount if enabled
	if min_send_amt > 0
		semi_cont_set = MOI.Semicontinuous(Float64(min_send_amt), Inf)
		for i in 1:N, j in 1:N, t in 1:T
			if !is_fixed(sent[i,j,t])
				delete_lower_bound(sent[i,j,t])
				@constraint(model, sent[i,j,t] in semi_cont_set)
			end
		end
	end

	# weight objective per-location by max load
	if severity_weighting
		max_load_null = [maximum(active_null[i,:] / beds[i]) for i in 1:N]
		severity_weight = [max_load_null[i] > 1.0 ? 0.0 : 9.0 for i in 1:N]
		add_to_expression!(objective, dot(sum(obj_dummy, dims=2), severity_weight))
	end

	if no_artificial_overflow
		for i in 1:N, t in 1:T
			if active_null[i,t] < beds[i]
				@constraint(model, active_patients[i,t] <= beds[i])
			end
		end
	end

	# penalize total sent if enabled
	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*sum(sent))
	end

	# penalize non-smoothness in sent patients if enabled
	if smoothness_penalty > 0
		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  (sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -(sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sent[:,:,1]))
	end

	# add setup costs if enabled
	if setup_cost > 0
		@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[i,j,:])+sum(sent[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy))
	end

	# enforce a minimum time between sending and receiving
	if sendreceive_switch_time > 0
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t]), sum(sent[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0])
		)
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sent[i,:,t])] in MOI.SOS1([1.0, 1.0])
		)
	end

	# load balancing
	if balancing_penalty > 0
		@variable(model, balancing_dummy[1:N,1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], balancing_dummy[i,t] >= (active_patients[i,t] / beds[i]) - balancing_thresh)
		add_to_expression!(objective, balancing_penalty * sum(balancing_dummy))
	end

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

##############################################
############### Block Model ##################
##############################################

function patient_block_allocation(
		beds::Array{:<Real,2},
		initial::Array{:<Real,2},
		discharged::Array{:<Real,3},
		admitted::Array{:<Real,3},
		los::Array{<:Any,1},
		adj_matrix::BitArray{2},
		transfer_graph::BitArray{2}
		bed_types::Array{Int,1};
		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		smoothness_penalty::Real=0,
		setup_cost::Real=0,
		sent_penalty::Real=0,
		balancing_thresh::Real=1.0,
		balancing_penalty::Real=0,
		severity_weighting::Bool=false,
		no_artificial_overflow::Bool=false,
		capacity_cushion::Real=0.0,
		verbose::Bool=false,
)
	B = size(beds,1)
	G, N, T = size(admitted)

	@assert size(beds) == (B,N)
	@assert size(initial) == (G,N)
	@assert size(discharged) == (G,N,T)
	@assert size(adj_matrix) == (N,N)
	@assert size(transfer_graph) == (G,G)
	@assert size(los) == (G,)
	@assert size(bed_types) == (G,)

	###############
	#### Setup ####
	###############

	F = zeros(Float64, G, T)
	f = zeros(Float64, G, T)
	for g in 1:G
		if isa(los[g], Int)
			F[g,:] = vcat(ones(Int, los[g]), zeros(Int, T-los[g]))
			f[g,los[g]+1] = 1.0
		elseif isa(los[g], Distribution)
			F[g,:] = 1.0 .- cdf.(los[g], 0:T-1)
			f[g,:] = pdf.(los[g], 0:T-1)
		else
			error("Invalid length of stay distribution")
		end
		f[g,:] = f[g,:] / sum(f[g,:])
	end

	if capacity_cushion > 0.0
		beds = beds .* (1.0 - capacity_cushion)
	end

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	###############
	## Variables ##
	###############

	@variable(model, sent[1:G,1:N,1:N,1:T] >= 0)
	@variable(model, obj_dummy[1:B,1:N,1:T] >= 0)

	#################
	## Expressions ##
	#################

	q(g,i,t) = @expression(model,
		admitted[g,i,t] + sum(l(g₁,i,t) for g₁ in findall(transfer_graph[:,g])) + sum(sent[g,j,i,t] - sent[g,i,j,t] for j in 1:N)
	)
	l(g,i,t) = @expression(model,
		discharged[g,i,t] + sum(f[g,t-t₁+1] * q(g,i,t₁) for t₁ in 1:t)
	)
	@expression(model, α[g=1:G,i=1:N,t=1:T],
		initial[g,i] + sum(sent[g,i,:,t]) + sum((q(g,i,t₁) - l(g,i,t₁)) for t₁ in 1:t)
	)

	q_null(g,i,t) = @expression(model,
		admitted[g,i,t] + sum(l_null(g₁,i,t) for g₁ in findall(transfer_graph[:,g]))
	)
	l_null(g,i,t) = @expression(model,
		discharged[g,i,t] + sum(f[g,t-t₁+1] * q_null(g,i,t₁) for t₁ in 1:t)
	)
	α_null = [(
		initial[g,i] + sum((q_null(g,i,t₁) - l_null(g,i,t₁)) for t₁ in 1:t)
		) for g in 1:G, i in 1:N, t in 1:T
	]

	# expression for the patient overflow
	@expression(model, overflow[b=1:B,i=1:N,t=1:T],
		sum(α[g,i,t] for g in bed_groups[b]) - beds[b,i]
	)

	objective = @expression(model, sum(obj_dummy))

	######################
	## Hard Constraints ##
	######################

	# ensure the number of active patients is non-negative
	@constraint(model, [g=1:G,i=1:N,t=1:T], α[g,i,t] >= 0)

	# only send new patients
	@constraint(model, [g=1:G,i=1:N,t=1:T], sum(sent[g,i,:,t]) <= admitted[g,i,t])

	# only send patients between connected locations
	for i in 1:N, j in 1:N
		if ~adj_matrix[i,j]
			for g in 1:G, t in 1:T
				fix(sent[g,i,j,t], 0, force=true)
			end
		end
	end

	# objective constraint
	@constraint(model, [b=1:B,i=1:N,t=1:T], obj_dummy[b,i,t] >= overflow[b,i,t])

	##########################
	## Optional Constraints ##
	##########################

	# enforce minimum transfer amount if enabled
	if min_send_amt > 0
		semi_cont_set = MOI.Semicontinuous(Float64(min_send_amt), Inf)
		for g in 1:G, i in 1:N, j in 1:N, t in 1:T
			if !is_fixed(sent[g,i,j,t])
				delete_lower_bound(sent[g,i,j,t])
				@constraint(model, sent[g,i,j,t] in semi_cont_set)
			end
		end
	end

	# weight objective per-location by max load
	if severity_weighting
		load_null = [(
			sum(active_null[g,i,t] for g in bed_groups[b]) / beds[b][i]
			) for b in bed_types, i in 1:N, t in 1:T
		]
		max_load_null = maximum(load_null, dims=3)[:,:,1]
		severity_weight = [max_load_null[b,i] > 1.0 ? 0.0 : 9.0 for b in 1:B, i in 1:N]

		add_to_expression!(objective, sum(sum(obj_dummy, dims=3) .* severity_weight))
	end

	if no_artificial_overflow
		for b in bed_types, i in 1:N, t in 1:T
			if (active_null[g,i,t] for g in bed_groups[b]) < beds[i]
				@constraint(model, sum(active_patients[g,i,t] for g in bed_groups[b]) <= beds[b][i])
			end
		end
	end

	# penalize total sent if enabled
	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*sum(sent))
	end

	# penalize non-smoothness in sent patients if enabled
	if smoothness_penalty > 0
		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  (sum(sent[:,:,:,t], dims=1) - sum(sent[:,:,:,t+1], dims=1))[1,:,:,:] .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -(sum(sent[:,:,:,t], dims=1) - sum(sent[:,:,:,t+1], dims=1))[1,:,:,:] .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sent[:,:,:,1]))
	end

	# add setup costs if enabled
	if setup_cost > 0
		@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[:,i,j,:])+sum(sent[:,j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy))
	end

	# enforce a minimum time between sending and receiving
	if sendreceive_switch_time > 0
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,:,i,t]), sum(sent[:,i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0])
		)
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,:,i,t:min(t+sendreceive_switch_time,T)]), sum(sent[:,i,:,t])] in MOI.SOS1([1.0, 1.0])
		)
	end

	# load balancing
	if balancing_penalty > 0
		@variable(model, balancing_dummy[bed_types,1:N,1:T] >= 0)
		@constraint(model, [b in bed_types,i=1:N,t=1:T],
			balancing_dummy[b,i,t] >= (sum(active_patients[g,i,t] for g in bed_groups[b]) / beds[b][i]) - balancing_thresh)
		add_to_expression!(objective, balancing_penalty * sum(balancing_dummy))
	end

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

end;
