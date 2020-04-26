module ReusableResourceAllocation

using JuMP
using Gurobi

using LinearAlgebra
using MathOptInterface

export patient_allocation, reusable_resource_allocation


function patient_allocation(
        beds::Array{Float32,1},
        initial_patients::Array{Float32,1},
        net_patients::Array{Float32,2},
        adj_matrix::BitArray{2};
        send_new_only::Bool=true,
        send_wait_period::Int=10,
		m::Real=1e-5,
		verbose::Bool=false,
)
	return reusable_resource_allocation(
		initial_patients,
		net_patients,
		repeat(beds, 1, size(net_patients, 2)),
		adj_matrix,
		objective=:overflow,
		send_new_only=send_new_only,
		send_wait_period=send_wait_period,
		m=m,
		verbose=verbose,
	)
end

function reusable_resource_allocation(
        initial_supply::Array{Float32,1},
        supply::Array{Float32,2},
		demand::Array{Float32,2},
        adj_matrix::BitArray{2};
		objective::Symbol=:shortage,
		send_new_only::Bool=true,
        send_wait_period::Int=10,
		m::Real=1e-5,
		verbose::Bool=false,
)
    N, T = size(supply)
    @assert(size(initial_supply, 1) == N)
	@assert(size(demand, 1) == N)
	@assert(size(demand, 2) == T)
	@assert(size(adj_matrix, 1) == N)
	@assert(size(adj_matrix, 2) == N)
	@assert(objective in [:shortage, :overflow])

    model = Model(Gurobi.Optimizer)
    if !verbose set_silent(model) end

    @variable(model, sent[1:N,1:N,1:T] >= 0)
    @variable(model, obj_dummy[1:N,1:T] >= 0)

    @objective(model, Min, sum(obj_dummy) + m*sum(sent))

	if send_new_only
        @constraint(model, [t=1:T],
            sum(sent[:,:,t], dims=2) .<= max.(0, supply[:,t])
        )
    else
	    @constraint(model, [t=1:T],
	        sum(sent[:,:,t], dims=2) .<=
	            initial_supply
	            .+ sum(supply[:,1:t], dims=2)
	            .- sum(sent[:,:,1:t-1], dims=[2,3])
	            .+ sum(sent[:,:,1:t-1], dims=[1,3])
	    )
	end

    for i = 1:N
        for j = 1:N
            if ~adj_matrix[i,j]
                @constraint(model, sum(sent[i,j,:]) .== 0)
            end
        end
    end

    if send_wait_period > 0
        @constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t]), sum(sent[i,:,t:min(t+send_wait_period,T)])] in MOI.SOS1([1.0, 1.0])
		)
    end

	flip_sign = (objective == :shortage) ? 1 : -1
	z1, z2 = (objective == :shortage) ? (0, -1) : (-1, 0)
    @constraint(model, [i=1:N,t=1:T],
        obj_dummy[i,t] >= flip_sign * (
			demand[i,t] - (
				initial_supply[i]
	            + sum(supply[i,1:t])
	            - sum(sent[i,:,1:t+z1])
	            + sum(sent[:,i,1:t+z2])
			)
		)
    )

    optimize!(model)
    return model
end


if abspath(PROGRAM_FILE) == @__FILE__
	N, T = 10, 14
	initial_supply = Float32.(rand(0:20, N))
	supply = Float32.(rand(0:2, N, T))
	demand = Float32.(rand(10:30, N, T))

	adj = rand(Bool, N, N)
	adj[tril(ones(Bool, N, N))] .= 0
	adj = BitArray(adj + adj')

	model = reusable_resource_allocation(initial_supply, supply, demand, adj, verbose=true)

	println("termination status: ", termination_status(model))
	println("solve time: ", round(solve_time(model), digits=3), "s")
	println("objective function value: ", round(objective_value(model), digits=3))

	sent = value.(model[:sent])
	total_shortage = sum(
		max(0,
			demand[i,t]
			- initial_supply[i]
			- sum(supply[i,1:t])
			+ sum(sent[i,:,1:t-1])
			- sum(sent[:,i,1:t])
		) for t in 1:T, i in 1:N
	)
	println("total sent: ", sum(sent))
	println("total shortage: ", total_shortage)
end

end;
