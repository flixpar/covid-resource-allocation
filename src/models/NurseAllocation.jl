module NurseAllocation

using JuMP
using Gurobi
using LinearAlgebra
using MathOptInterface

export nurse_allocation


function nurse_allocation(
		initial_nurses::Array{<:Real,1},
		demand::Array{<:Real,2},
		adj_matrix::BitArray{2};

		sent_penalty::Real=0,
		smoothness_penalty::Real=0,

		no_artificial_shortage::Bool=false,
		no_worse_shortage::Bool=false,
		fully_connected::Bool=false,

		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		setup_cost::Real=0,

		verbose::Bool=false,
)

	###############
	#### Setup ####
	###############

	N, T = size(demand)
	@assert(size(initial_nurses) == (N,))
	@assert(size(adj_matrix) == (N,N))

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

	# compute active nurses
	@expression(model, active_nurses[i=1:N,t=0:T],
		initial_nurses[i]
		- sum(sent[i,:,1:t])
		+ sum(sent[:,i,1:t])
	)

	objective = @expression(model, sum(obj_dummy))

	######################
	## Hard Constraints ##
	######################

	# sent nurses ≦ active nurses
	@constraint(model, [i=1:N,t=1:T], sum(sent[i,:,t]) <= active_nurses[i,t-1])

	# objective
	@constraint(model, [i=1:N,t=1:T], obj_dummy[i,t] >= demand[i,t] - active_nurses[i,t])

	for i in 1:N, t in 1:T
		fix(sent[i,i,t], 0, force=true)
	end
	if !fully_connected
		for i in 1:N, j in 1:N
			if ~adj_matrix[i,j]
				for t in 1:T
					fix(sent[i,j,t], 0, force=true)
				end
			end
		end
	end

	##########################
	## Optional Constraints ##
	##########################

	if no_artificial_shortage
		for i in 1:N, t in 1:T
			if initial_nurses[i] >= demand[i,t]
				@constraint(model, active_nurses[i,t] >= demand[i,t])
			end
		end
	end

	if no_worse_shortage
		for i in 1:N, t in 1:T
			if initial_nurses[i] < demand[i,t]
				@constraint(model, active_nurses[i,t] >= initial_nurses[i])
			end
		end
	end

	if min_send_amt <= 0
		@constraint(model, sent .>= 0)
	else
		@constraint(model, [i=1:N,j=1:N,t=1:T], sent[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
	end

	if sendreceive_switch_time > 0
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t]), sum(sent[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0])
		)
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sent[i,:,t])] in MOI.SOS1([1.0, 1.0])
		)
	end

	########################
	## Optional Penalties ##
	########################

	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*sum(sent))
	end

	if smoothness_penalty > 0
		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  (sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -(sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sent[:,:,1]))
	end

	if setup_cost > 0
		@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[i,j,:])+sum(sent[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy))
	end

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

end;
