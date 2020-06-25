module NurseAllocation

using JuMP
using Gurobi

using LinearAlgebra
using MathOptInterface

export nurse_allocation


function nurse_allocation(
        initial_nurses::Array{Float32,1},
		demand::Array{Float32,2},
        adj_matrix::BitArray{2};
        sendreceive_switch_time::Int=0,
		fully_connected::Bool=false,
		min_send_amt::Real=0,
		smoothness_penalty::Real=0,
		setup_cost::Real=0,
		sent_penalty::Real=0,
		target_load::Real=1.25,
		target_load_gap::Real=0.25,
		load_penalty::Real=10.0,
		disallow_artifical_shortage::Bool=false,
		verbose::Bool=false,
)
    N, T = size(demand)
    @assert(size(initial_nurses) == (N,))
	@assert(size(adj_matrix) == (N,N))

    model = Model(Gurobi.Optimizer)
    if !verbose set_silent(model) end

	@variable(model, sent[1:N,1:N,1:T])
    @variable(model, obj_dummy[1:N,1:T] >= 0)

	if min_send_amt <= 0
		@constraint(model, sent .>= 0)
	else
		@constraint(model, [i=1:N,j=1:N,t=1:T], sent[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
	end

	objective = @expression(model, sum(obj_dummy))

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
	@objective(model, Min, objective)

	if fully_connected
		for i in 1:N
			@constraint(model, sent[i,i,:] .== 0)
		end
	else
	    for i = 1:N
	        for j = 1:N
	            if ~adj_matrix[i,j]
	                @constraint(model, sent[i,j,:] .== 0)
	            end
	        end
	    end
	end

    if sendreceive_switch_time > 0
        @constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t]), sum(sent[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0])
		)
		@constraint(model, [i=1:N,t=1:T-1],
            [sum(sent[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sent[i,:,t])] in MOI.SOS1([1.0, 1.0])
        )
    end

	# compute active nurses
	@expression(model, active_nurses[i=1:N,t=0:T],
		initial_nurses[i]
		- sum(sent[i,:,1:t])
		+ sum(sent[:,i,1:t])
	)

	# sent nurses ≦ active nurses
	@constraint(model, [i=1:N,t=1:T], sum(sent[i,:,t]) <= active_nurses[i,t-1])

	# objective
	@constraint(model, [i=1:N,t=1:T], obj_dummy[i,t] >= demand[i,t] - active_nurses[i,t])

	if disallow_artifical_shortage
		m = 1e-5
		@variable(model, has_shortage[i=1:N,t=1:T], Bin)
		@constraint(model, [i=1:N,t=1:T], m*(demand[i,t] - active_nurses[i,t]) <= has_shortage[i,t])
		@constraint(model, [i=1:N,t=1:T], 1 + m*(demand[i,t] - active_nurses[i,t]) >= has_shortage[i,t])
		@constraint(model, [i=1:N,t=1:T], has_shortage[i,t] => {active_nurses[i,t] >= initial_nurses[i]})
	end

	if load_penalty > 0
		@variable(model, load_dummy_abs[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T],  (active_nurses[i,t] - target_load*demand[i,t]) <= load_dummy_abs[i,t])
		@constraint(model, [i=1:N,t=1:T], -(active_nurses[i,t] - target_load*demand[i,t]) <= load_dummy_abs[i,t])

		@variable(model, load_dummy[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], load_dummy[i,t] >= load_dummy_abs[i,t] - target_load_gap*demand[i,t])
		add_to_expression!(objective, load_penalty * sum(load_dummy))
	end

    optimize!(model)
    return model
end

end;
