module PatientAllocation

using JuMP
using Gurobi

using LinearAlgebra
using MathOptInterface

export patient_allocation


##############################################
############# Standard Model #################
##############################################

function patient_allocation(
		beds::Array{<:Real,1},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		adj_matrix::BitArray{2};
		hospitalized_days::Int=8,
		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		smoothness_penalty::Real=0,
		setup_cost::Real=0,
		sent_penalty::Real=0,
		balancing_thresh::Real=1.0,
		balancing_penalty::Real=0,
		severity_weighting::Bool=false,
		verbose::Bool=false,
)
	N, T = size(admitted_patients)
	@assert(size(initial_patients, 1) == N)
	@assert(size(beds, 1) == N)
	@assert(size(adj_matrix) == (N,N))
	@assert(size(discharged_patients) == (N, T))

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

	# expression for the number of active patients
	ts(t) = max(1, t - hospitalized_days + 1)
	@expression(model, active_patients[i=1:N,t=0:T],
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(admitted_patients[i,ts(t):t])
		- sum(sent[i,:,ts(t):t])
		+ sum(sent[:,i,ts(t):t])
	)

	# expression for the patient overflow
	@expression(model, overflow[i=1:N,t=1:T], active_patients[i,t] + sum(sent[i,:,t]) - beds[i])

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
		active_null = [(
				initial_patients[i]
				- sum(discharged_patients[i,1:t])
				+ sum(admitted_patients[i,max(1,t-hospitalized_days+1):t])
			) for i in 1:N, t in 1:T
		]
		max_load_null = [maximum(active_null[i,:] / beds[i]) for i in 1:N]
		severity_weight = [max_load_null[i] > 1 ? 1.0 : 10.0 for i in 1:N]

		add_to_expression!(objective, dot(sum(obj_dummy, dims=2), (severity_weight .- 1.0)))
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

end;
