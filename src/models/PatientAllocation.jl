module PatientAllocation

using JuMP
using Gurobi

using LinearAlgebra
using MathOptInterface
using Distributions
using Memoize
using Statistics


##############################################
############# Standard Model #################
##############################################

function patient_redistribution(
		capacity::Array{<:Real},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		adj_matrix::BitArray{2}, los::Union{<:Distribution,Array{<:Real,1},Int};

		capacity_cushion::Real=-1, capacity_weights::Array{<:Real,1}=Int[],
		no_artificial_overflow::Bool=false, no_worse_overflow::Bool=false,
		sent_penalty::Real=0, smoothness_penalty::Real=0,

		sendreceive_gap::Int=0, min_send_amt::Real=0,
		balancing_thresh::Real=1.0, balancing_penalty::Real=0,
		severity_weighting::Bool=false, setup_cost::Real=0,

		verbose::Bool=false,
	)

	###############
	#### Setup ####
	###############

	if ndims(capacity) == 1
		capacity = reshape(capacity, (:,1))
	end

	N, T = size(admitted_patients)
	C = size(capacity, 2)
	check_sizes(initial_patients, discharged_patients, admitted_patients, capacity)

	if 0 < capacity_cushion < 1
		capacity = capacity .* (1.0 - capacity_cushion)
	end

	if isempty(capacity_weights)
		capacity_weights = ones(Int, C)
	end

	L = discretize_los(los, T)

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	###############
	## Variables ##
	###############

	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, overflow[1:N,1:T,1:C] >= 0)

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
	active_null = compute_active_null(initial_patients, discharged_patients, admitted_patients, L)

	# objective function
	objective = @expression(model, dot(capacity_weights, sum(overflow, dims=(1,2))))

	######################
	## Hard Constraints ##
	######################

	# ensure the number of active patients is non-negative
	@constraint(model, [i=1:N,t=1:T], active_patients[i,t] >= 0)

	# only send new patients
	@constraint(model, [t=1:T], sum(sent[:,:,t], dims=2) .<= admitted_patients[:,t])

	# objective constraint
	@constraint(model, [i=1:N,t=1:T,c=1:C], overflow[i,t,c] >= active_patients[i,t] - capacity[i,c])

	################################
	## Optional Constraints/Costs ##
	################################

	enforce_adj!(model, sent, adj_matrix)
	enforce_no_artificial_overflow!(model, no_artificial_overflow, active_patients, active_null, capacity)
	enforce_no_worse_overflow!(model, no_worse_overflow, active_patients, active_null, capacity)
	enforce_minsendamt!(model, sent, min_send_amt)
	enforce_sendreceivegap!(model, sent, sendreceive_gap)

	add_sent_penalty!(model, sent, objective, sent_penalty)
	add_smoothness_penalty!(model, sent, objective, smoothness_penalty)
	add_setup_cost!(model, sent, objective, setup_cost)
	add_loadbalancing_penalty!(model, sent, objective, balancing_penalty, balancing_thresh, active_patients, capacity)
	add_severity_weighting!(model, sent, objective, severity_weighting, overflow, active_null, capacity)

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

##############################################
############### Load Balance #################
##############################################

function patient_loadbalance(
		capacity::Array{<:Real},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		adj_matrix::BitArray{2}, los::Union{<:Distribution,Array{<:Real,1},Int};

		capacity_cushion::Real=-1, capacity_weights::Array{<:Real,1}=Int[],
		no_artificial_overflow::Bool=false, no_worse_overflow::Bool=false,
		sent_penalty::Real=0, smoothness_penalty::Real=0,

		sendreceive_gap::Int=0, min_send_amt::Real=0,
		setup_cost::Real=0,

		verbose::Bool=false,
	)

	###############
	#### Setup ####
	###############

	if ndims(capacity) == 1
		capacity = reshape(capacity, (:,1))
	end

	N, T = size(admitted_patients)
	C = size(capacity, 2)
	check_sizes(initial_patients, discharged_patients, admitted_patients, capacity)

	if 0 < capacity_cushion < 1
		capacity = capacity .* (1.0 - capacity_cushion)
	end

	if isempty(capacity_weights)
		capacity_weights = ones(Int, C)
	end

	L = discretize_los(los, T)

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	###############
	## Variables ##
	###############

	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, load_objective[1:N,1:T,1:C] >= 0)

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
	active_null = compute_active_null(initial_patients, discharged_patients, admitted_patients, L)

	# expression for the patient load
	@expression(model, load[i=1:N,t=1:T,c=1:C], active_patients[i,t] / capacity[i,c])

	# objective function
	objective = @expression(model, dot(capacity_weights, sum(load_objective, dims=(1,2))))

	######################
	## Hard Constraints ##
	######################

	# ensure the number of active patients is non-negative
	@constraint(model, [i=1:N,t=1:T], active_patients[i,t] >= 0)

	# only send new patients
	@constraint(model, [t=1:T], sum(sent[:,:,t], dims=2) .<= admitted_patients[:,t])

	# objective constraint
	@constraint(model, [i=1:N,t=1:T,c=1:C],  (load[i,t,c] - mean(load[:,t,c])) <= load_objective[i,t,c])
	@constraint(model, [i=1:N,t=1:T,c=1:C], -(load[i,t,c] - mean(load[:,t,c])) <= load_objective[i,t,c])

	################################
	## Optional Constraints/Costs ##
	################################

	enforce_adj!(model, sent, adj_matrix)
	enforce_no_artificial_overflow!(model, no_artificial_overflow, active_patients, active_null, capacity)
	enforce_no_worse_overflow!(model, no_worse_overflow, active_patients, active_null, capacity)
	enforce_minsendamt!(model, sent, min_send_amt)
	enforce_sendreceivegap!(model, sent, sendreceive_gap)

	add_sent_penalty!(model, sent, objective, sent_penalty)
	add_smoothness_penalty!(model, sent, objective, smoothness_penalty)
	add_setup_cost!(model, sent, objective, setup_cost)

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

##############################################
############ Surge-Level Model ###############
##############################################

function patient_surgelevel_redistribution(
		capacity::Array{<:Real,2},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		adj_matrix::BitArray{2}, los::Union{<:Distribution,Array{<:Real,1},Int};

		capacity_cushion::Real=-1, capacity_weights::Array{<:Real,1}=Int[],
		no_artificial_overflow::Bool=false, no_worse_overflow::Bool=false,
		sent_penalty::Real=0, smoothness_penalty::Real=0,

		sendreceive_gap::Int=0, min_send_amt::Real=0,
		balancing_thresh::Real=1.0, balancing_penalty::Real=0,
		setup_cost::Real=0,

		mipgap::Real=0.1, timelimit::Real=240, M::Real=1e6,

		verbose::Bool=false,
	)

	###############
	#### Setup ####
	###############

	N, T = size(admitted_patients)
	C = size(capacity, 2)
	check_sizes(initial_patients, discharged_patients, admitted_patients, capacity)

	if 0 < capacity_cushion < 1
		capacity = capacity .* (1.0 - capacity_cushion)
	end

	if isempty(capacity_weights)
		capacity_weights = ones(Int, C)
	end

	L = discretize_los(los, T)

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	set_optimizer_attribute(model, "MIPGap", mipgap)
	set_optimizer_attribute(model, "TimeLimit", timelimit)

	###############
	## Variables ##
	###############

	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, surgelevel[1:N,1:T,1:C], Bin)

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
	active_null = compute_active_null(initial_patients, discharged_patients, admitted_patients, L)

	# expression for the patient overflow
	@expression(model, overflow[i=1:N,t=1:T,c=1:C], active_patients[i,t] - capacity[i,c])

	# weights
	weights = hcat(capacity[:,1], diff(capacity, dims=2))
	weights = weights .* permutedims(repeat(capacity_weights, 1, N), (2,1))

	# objective function
	objective = @expression(model, sum(sum(surgelevel, dims=2)[:,1,:] .* weights))

	######################
	## Hard Constraints ##
	######################

	# ensure the number of active patients is non-negative
	@constraint(model, [i=1:N,t=1:T], active_patients[i,t] >= 0)

	# only send new patients
	@constraint(model, [t=1:T], sum(sent[:,:,t], dims=2) .<= admitted_patients[:,t])

	# set surgelevel
	for i in 1:N, t in 1:T, c in 1:C
		@constraint(model, M * surgelevel[i,t,c] >= overflow[i,t,c])
		@constraint(model, M * (1 - surgelevel[i,t,c]) >= -overflow[i,t,c])
	end

	################################
	## Optional Constraints/Costs ##
	################################

	enforce_adj!(model, sent, adj_matrix)
	enforce_no_artificial_overflow!(model, no_artificial_overflow, active_patients, active_null, capacity)
	enforce_no_worse_overflow!(model, no_worse_overflow, active_patients, active_null, capacity)
	enforce_minsendamt!(model, sent, min_send_amt)
	enforce_sendreceivegap!(model, sent, sendreceive_gap)

	add_sent_penalty!(model, sent, objective, sent_penalty)
	add_smoothness_penalty!(model, sent, objective, smoothness_penalty)
	add_setup_cost!(model, sent, objective, setup_cost)
	add_loadbalancing_penalty!(model, sent, objective, balancing_penalty, balancing_thresh, active_patients, capacity)

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

##############################################
####### Surge-Level Load Balancing ###########
##############################################

function patient_surgelevel_loadbalance(
		capacity::Array{<:Real},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		adj_matrix::BitArray{2}, los::Union{<:Distribution,Array{<:Real,1},Int};

		capacity_cushion::Real=-1, capacity_weights::Array{<:Real,1}=Int[],
		no_artificial_overflow::Bool=false, no_worse_overflow::Bool=false,
		sent_penalty::Real=0, smoothness_penalty::Real=0,

		sendreceive_gap::Int=0, min_send_amt::Real=0,
		setup_cost::Real=0,

		mipgap::Real=0.1, timelimit::Real=240, M::Real=1e6,
		verbose::Bool=false,
	)

	###############
	#### Setup ####
	###############

	N, T = size(admitted_patients)
	C = size(capacity, 2)
	check_sizes(initial_patients, discharged_patients, admitted_patients, capacity)

	if 0 < capacity_cushion < 1
		capacity = capacity .* (1.0 - capacity_cushion)
	end

	if isempty(capacity_weights)
		capacity_weights = ones(Int, C)
	end

	L = discretize_los(los, T)

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	set_optimizer_attribute(model, "MIPGap", mipgap)
	set_optimizer_attribute(model, "TimeLimit", timelimit)

	###############
	## Variables ##
	###############

	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, surgelevel[1:N,1:T,1:C], Bin)
	@variable(model, load_obj[1:N,1:T] >= 0)

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
	active_null = compute_active_null(initial_patients, discharged_patients, admitted_patients, L)

	# expression for the patient overflow
	@expression(model, overflow[i=1:N,t=1:T,c=1:C], active_patients[i,t] - capacity[i,c])

	# objective function
	objective = @expression(model, dot(capacity_weights, sum(load_obj, dims=(1,2))))

	######################
	## Hard Constraints ##
	######################

	# ensure the number of active patients is non-negative
	@constraint(model, [i=1:N,t=1:T], active_patients[i,t] >= 0)

	# only send new patients
	@constraint(model, [t=1:T], sum(sent[:,:,t], dims=2) .<= admitted_patients[:,t])

	# set surge-level
	for i in 1:N, t in 1:T, c in 1:C
		@constraint(model, M * surgelevel[i,t,c] >= overflow[i,t,c])
		@constraint(model, M * (1 - surgelevel[i,t,c]) >= -overflow[i,t,c])
	end

	# load balancing objective
	@constraint(model, [i=1:N,t=1:T],  (sum(surgelevel[i,t,:]) - mean(sum(surgelevel[:,t,:], dims=2))) <= load_obj[i,t])
	@constraint(model, [i=1:N,t=1:T], -(sum(surgelevel[i,t,:]) - mean(sum(surgelevel[:,t,:], dims=2))) <= load_obj[i,t])

	################################
	## Optional Constraints/Costs ##
	################################

	enforce_adj!(model, sent, adj_matrix)
	enforce_no_artificial_overflow!(model, no_artificial_overflow, active_patients, active_null, capacity)
	enforce_no_worse_overflow!(model, no_worse_overflow, active_patients, active_null, capacity)
	enforce_minsendamt!(model, sent, min_send_amt)
	enforce_sendreceivegap!(model, sent, sendreceive_gap)

	add_sent_penalty!(model, sent, objective, sent_penalty)
	add_smoothness_penalty!(model, sent, objective, smoothness_penalty)
	add_setup_cost!(model, sent, objective, setup_cost)

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
		beds::Array{<:Real,2},
		initial::Array{<:Real,2},
		discharged::Array{<:Real,3},
		admitted::Array{<:Real,3},
		los::Array{<:Any,1},
		adj_matrix::BitArray{2},
		transfer_graph::BitArray{2},
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
		no_worse_overflow::Bool=false,
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

	bed_groups = [sort(findall(x -> x == b, bed_types)) for b in 1:B]

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

	@memoize q(g,i,t) = @expression(model,
		admitted[g,i,t] + tr(g,i,t) + sum(sent[g,j,i,t] - sent[g,i,j,t] for j in 1:N)
	)
	@memoize l(g,i,t) = @expression(model,
		discharged[g,i,t] + sum(f[g,t-t₁+1] * q(g,i,t₁) for t₁ in 1:t)
	)
	@memoize tr(g,i,t) = @expression(model,
		any(transfer_graph[:,g]) ?
			sum(l(g₁,i,t) for g₁ in findall(transfer_graph[:,g]))
			: 0
	)
	@expression(model, α[g=1:G,i=1:N,t=1:T],
		initial[g,i] + sum(sent[g,i,:,t]) + sum((q(g,i,t₁) - l(g,i,t₁)) for t₁ in 1:t)
	)

	@memoize q_null(g,i,t) = (
		admitted[g,i,t] + tr_null(g,i,t)
	)
	@memoize l_null(g,i,t) = (
		discharged[g,i,t] + sum(f[g,t-t₁+1] * q_null(g,i,t₁) for t₁ in 1:t)
	)
	@memoize tr_null(g,i,t) = (
		any(transfer_graph[:,g]) ?
			sum(l_null(g₁,i,t) for g₁ in findall(transfer_graph[:,g]))
			: 0
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

	# only transfer patients within groups with incoming patients
	for g in findall(sum(admitted, dims=[2,3])[:] .== 0)
		for i in 1:N, j in 1:N, t in 1:T
			fix(sent[g,i,j,t], 0, force=true)
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
			sum(α_null[g,i,t] for g in bed_groups[b]) / beds[b,i]
			) for b in bed_types, i in 1:N, t in 1:T
		]
		max_load_null = maximum(load_null, dims=3)[:,:,1]
		severity_weight = [max_load_null[b,i] > 1.0 ? 0.0 : 9.0 for b in 1:B, i in 1:N]

		add_to_expression!(objective, sum(sum(obj_dummy, dims=3) .* severity_weight))
	end

	if no_artificial_overflow
		for b in bed_types, i in 1:N, t in 1:T
			if sum(α_null[g,i,t] for g in bed_groups[b]) < beds[b,i]
				@constraint(model, sum(α[g,i,t] for g in bed_groups[b]) <= beds[b,i])
			end
		end
	end

	if no_worse_overflow
		for b in bed_types, i in 1:N, t in 1:T
			if sum(α_null[g,i,t] for g in bed_groups[b]) > beds[i]
				@constraint(model, sum(α[g,i,t] for g in bed_groups[b]) <= sum(α_null[g,i,t] for g in bed_groups[b]))
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
			balancing_dummy[b,i,t] >= (sum(α[g,i,t] for g in bed_groups[b]) / beds[b,i]) - balancing_thresh)
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
############### Robust Model #################
##############################################

function patient_redistribution_robust(
		capacity::Array{<:Real},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients_nominal::Array{<:Real,2},
		admitted_patients_lb::Array{<:Real,2},
		admitted_patients_ub::Array{<:Real,2},
		adj_matrix::BitArray{2}, los::Union{<:Distribution,Array{<:Real,1},Int};

		Γ::Int=3,
		capacity_cushion::Real=-1, capacity_weights::Array{<:Real,1}=Int[],
		no_artificial_overflow::Bool=false, no_worse_overflow::Bool=false,
		sent_penalty::Real=0, smoothness_penalty::Real=0,
		sendreceive_gap::Int=0, min_send_amt::Real=0, setup_cost::Real=0,

		verbose::Bool=false,
	)

	###############
	#### Setup ####
	###############

	if ndims(capacity) == 1
		capacity = reshape(capacity, (:,1))
	end

	N, T = size(admitted_patients_nominal)
	C = size(capacity, 2)
	check_sizes(initial_patients, discharged_patients, admitted_patients_nominal, capacity)

	if 0 < capacity_cushion < 1
		capacity = capacity .* (1.0 - capacity_cushion)
	end

	if isempty(capacity_weights)
		capacity_weights = ones(Int, C)
	end

	L = discretize_los(los, T)

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	###############
	## Variables ##
	###############

	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, overflow[1:N,1:T,1:C] >= 0)

	#################
	## Expressions ##
	#################

	# objective function
	objective = @expression(model, dot(capacity_weights, sum(overflow, dims=(1,2))))

	active_null_ub = [(
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(L[t-t₁+1] * (t₁ > (t-Γ) ? admitted_patients_ub[i,t₁] : admitted_patients_nominal[i,t₁]) for t₁ in 1:t)
	) for i in 1:N, t in 1:T]

	active_null_lb = [(
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(L[t-t₁+1] * (t₁ > (t-Γ) ? admitted_patients_lb[i,t₁] : admitted_patients_nominal[i,t₁]) for t₁ in 1:t)
	) for i in 1:N, t in 1:T]

	######################
	## Hard Constraints ##
	######################

	# ensure the number of active patients is non-negative
	@constraint(model, [i=1:N,t=1:T], 0 <=
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(L[t-t₁+1] * (
			(t₁ > (t-Γ) ? admitted_patients_ub[i,t₁] : admitted_patients_nominal[i,t₁])
			- sum(sent[i,:,t₁])
			+ sum(sent[:,i,t₁])
		) for t₁ in 1:t)
		+ sum(sent[i,:,t])
	)

	# only send new patients
	@constraint(model, [i=1:N,t=1:T], sum(sent[i,:,t]) <= admitted_patients_lb[i,t])

	# objective constraint
	@constraint(model, [i=1:N,t=1:T,c=1:C], overflow[i,t,c] >=
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(L[t-t₁+1] * (t₁ > (t-Γ) ? admitted_patients_ub[i,t₁] : admitted_patients_nominal[i,t₁]) for t₁ in 1:t)
		- sum(L[t-t₁+1] * sum(sent[i,:,t₁]) for t₁ in 1:t)
		+ sum(L[t-t₁+1] * sum(sent[:,i,t₁]) for t₁ in 1:t)
		+ sum(sent[i,:,t])
		- capacity[i,c]
	)

	##########################
	## Optional Constraints ##
	##########################

	enforce_adj!(model, sent, adj_matrix)
	enforce_minsendamt!(model, sent, min_send_amt)
	enforce_sendreceivegap!(model, sent, sendreceive_gap)

	add_sent_penalty!(model, sent, objective, sent_penalty)
	add_smoothness_penalty!(model, sent, objective, smoothness_penalty)
	add_setup_cost!(model, sent, objective, setup_cost)

	if no_artificial_overflow
		for i in 1:N, t in 1:T
			if active_null_ub[i,t] < capacity[i,end]
				@constraint(model, capacity[i,end] >=
					initial_patients[i]
					- sum(discharged_patients[i,1:t])
					+ sum(L[t-t₁+1] * (t₁ > (t-Γ) ? admitted_patients_ub[i,t₁] : admitted_patients_nominal[i,t₁]) for t₁ in 1:t)
					- sum(L[t-t₁+1] * sum(sent[i,:,t₁]) for t₁ in 1:t)
					+ sum(L[t-t₁+1] * sum(sent[:,i,t₁]) for t₁ in 1:t)
					+ sum(sent[i,:,t])
				)
			end
		end
	end

	if no_worse_overflow
		for i in 1:N, t in 1:T
			if active_null_lb[i,t] > capacity[i,end]
				@constraint(model, 0 >=
					- sum(L[t-t₁+1] * sum(sent[i,:,t₁]) for t₁ in 1:t)
					+ sum(L[t-t₁+1] * sum(sent[:,i,t₁]) for t₁ in 1:t)
					+ sum(sent[i,:,t])
				)
			end
		end
	end

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

##############################################
############# Helper Functions ###############
##############################################

function discretize_los(los, T)
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
	return L
end

function compute_active_null(initial_patients, discharged_patients, admitted_patients, L)
	N, T = size(admitted_patients)
	active_null = [(
			initial_patients[i]
			- sum(discharged_patients[i,1:t])
			+ sum(L[t-t₁+1] * admitted_patients[i,t₁] for t₁ in 1:t)
		) for i in 1:N, t in 1:T
	]
	return active_null
end

function check_sizes(initial_patients, discharged_patients, admitted_patients, beds)
	N, T = size(admitted_patients)
	@assert(size(initial_patients) == (N,))
	@assert(size(discharged_patients) == (N, T))
	@assert(size(beds, 1) == N)
	return
end

##############################################
########### Optional Constraints #############
##############################################

# only send patients between connected locations
function enforce_adj!(model, sent, adj_matrix)
	N, _, T = size(sent)
	@assert(size(adj_matrix) == (N,N))
	for i in 1:N, j in 1:N
		if ~adj_matrix[i,j]
			for t in 1:T
				fix(sent[i,j,t], 0, force=true)
			end
		end
	end
	return
end

function enforce_no_artificial_overflow!(model, no_artificial_overflow, active_patients, active_null, capacity)
	if no_artificial_overflow
		N, T = size(active_null)
		for i in 1:N, t in 1:T
			if active_null[i,t] <= capacity[i,end]
				@constraint(model, active_patients[i,t] <= capacity[i,end])
			end
		end
	end
	return
end

function enforce_no_worse_overflow!(model, no_worse_overflow, active_patients, active_null, capacity)
	if no_worse_overflow
		N, T = size(active_null)
		for i in 1:N, t in 1:T
			if active_null[i,t] >= capacity[i,end]
				@constraint(model, active_patients[i,t] <= active_null[i,t])
			end
		end
	end
	return
end

# enforce minimum transfer amount if enabled
function enforce_minsendamt!(model, sent, min_send_amt)
	if min_send_amt > 0
		N, _, T = size(sent)
		semi_cont_set = MOI.Semicontinuous(Float64(min_send_amt), Inf)
		for i in 1:N, j in 1:N, t in 1:T
			if !is_fixed(sent[i,j,t])
				delete_lower_bound(sent[i,j,t])
				@constraint(model, sent[i,j,t] in semi_cont_set)
			end
		end
	end
	return
end

# enforce a minimum time between sending and receiving
function enforce_sendreceivegap!(model, sent, sendreceive_gap)
	if sendreceive_gap > 0
		N, _, T = size(sent)
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t]), sum(sent[i,:,t:min(t+sendreceive_gap,T)])] in MOI.SOS1([1.0, 1.0])
		)
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t:min(t+sendreceive_gap,T)]), sum(sent[i,:,t])] in MOI.SOS1([1.0, 1.0])
		)
	end
	return
end

##############################################
############ Optional Penalties ##############
##############################################

# penalize total sent if enabled
function add_sent_penalty!(model, sent, objective, sent_penalty)
	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*sum(sent))
	end
	return
end

# penalize non-smoothness in sent patients if enabled
function add_smoothness_penalty!(model, sent, objective, smoothness_penalty)
	if smoothness_penalty > 0
		N, _, T = size(sent)

		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  (sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -(sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sent[:,:,1]))
	end
	return
end

# add setup costs if enabled
function add_setup_cost!(model, sent, objective, setup_cost)
	if setup_cost > 0
		N, _, T = size(sent)
		@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[i,j,:])+sum(sent[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy))
	end
	return
end

# load balancing penalty
function add_loadbalancing_penalty!(model, sent, objective, balancing_penalty, balancing_thresh, active_patients, capacity)
	if balancing_penalty > 0
		N, _, T = size(sent)
		@variable(model, balancing_dummy[1:N,1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], balancing_dummy[i,t] >= (active_patients[i,t] / capacity[i,1]) - balancing_thresh)
		add_to_expression!(objective, balancing_penalty * sum(balancing_dummy))
	end
	return
end

# weight objective per-location by max load
function add_severity_weighting!(model, sent, objective, severity_weighting, overflow, active_null, capacity)
	if severity_weighting
		N, _, T = size(sent)
		max_load_null = [maximum(active_null[i,:] / capacity[i,1]) for i in 1:N]
		severity_weight = [max_load_null[i] > 1.0 ? 0.0 : 9.0 for i in 1:N]
		add_to_expression!(objective, dot(sum(overflow, dims=2), severity_weight))
	end
	return
end

end;
