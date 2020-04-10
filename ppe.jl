using JuMP
using Gurobi

function ppe_model_optimize(initial_resource::Array{Int,1}, needs::Array{Int,2}, growth::Array{Int,1})
	N, T = size(needs)

	@assert(size(initial_resource, 1) == N)
	@assert(size(growth, 1) == T)

	model = Model(Gurobi.Optimizer)
	@variable(model, used[1:N,1:T] >= 0)
	@variable(model, sent[1:N,1:T] >= 0)
	@variable(model, recieved[1:N,1:T] >= 0)

	@objective(model, Max, sum(used))

	# amount used at hospital i at time t ≦ amount available at hospital i at time t
	@constraint(model, [t=1:T],
		used[:,t] .<=
			initial_resource
			- sum(used[:,1:t-1], dims=2)
			+ sum(recieved[:,1:t], dims=2)
			- sum(sent[:,1:t], dims=2)
	)

	# amount sent for hospital i at time t ≦ amount available at hospital i at time t
	@constraint(model, [t=1:T],
		sent[:,t] .<=
			initial_resource
			- sum(used[:,1:t-1], dims=2)
			+ sum(recieved[:,1:t-1], dims=2)
			- sum(sent[:,1:t-1], dims=2)
	)

	# can't use more than need
	@constraint(model, used .<= needs)

	# total amount recieved ≦ total amount sent plus total amount stockpiled ∀ t
	@constraint(model, [t=1:T], sum(recieved[:,t]) <= sum(sent[:,t]) + sum(growth[1:t]) + sum(sent[:,1:t-1]) - sum(recieved[:,1:t-1]))

	optimize!(model)
	return model
end

function ppe_graph_model_optimize(initial_resource::Array{Int,1}, needs::Array{Int,2}, growth::Array{Int,1})
	N, T = size(needs)

	@assert(size(initial_resource, 1) == N)
	@assert(size(growth, 1) == T)

	model = Model(Gurobi.Optimizer)
	@variable(model, used[1:N,1:T] >= 0)
	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, sent_stockpile[1:N,1:T] >= 0)

	@objective(model, Max, sum(used))

	@constraint(model, [t=1:T],
		used[:,t] .<=
			initial_resource                      #   amount at start
			- sum(used[:,1:t-1], dims=2)          # - amount used so far
			+ sum(sent[:,:,1:t], dims=[1,3])      # + amount sent to i so far
			- sum(sent[:,:,1:t], dims=[2,3])      # - amount sent from i so far
			+ sum(sent_stockpile[:,1:t], dims=2)  # + amount sent from stockpile so far
	)
	@constraint(model, [t=1:T],
		sum(sent[:,:,t], dims=2) .<=
			initial_resource                        #   amount at start
			- sum(used[:,1:t-1], dims=2)            # - amount used so far
			+ sum(sent[:,:,1:t-1], dims=[1,3])      # + amount sent to i so far
			- sum(sent[:,:,1:t-1], dims=[2,3])      # - amount sent from i so far
			+ sum(sent_stockpile[:,1:t-1], dims=2)  # + amount sent from stockpile so far
	)
	@constraint(model, used .<= needs)
	@constraint(model, [i=1:N], sent[i,i,:] .== 0)  # nothing sent to self
	@constraint(model, [t=1:T], sum(sent_stockpile[:,t]) <= sum(growth[1:t]) - sum(sent_stockpile[:,1:t-1]))

	optimize!(model)
	return model
end

if abspath(PROGRAM_FILE) == @__FILE__
	N, T = 20, 60
	initial_resource = rand(1:250, N)
	needs = rand(1:50, N, T)
	growth = rand(0:100, T)

	model = ppe_model_optimize(initial_resource, needs, growth)

	used = value.(model[:used])
	shortage = sum(needs .- used)
	println("Shortage: ", shortage)

	total = sum(initial_resource)
	shortage = 0
	for t = 1:T
		global total += growth[t]
		used = min(sum(needs[:,t]), total)
		global shortage += sum(needs[:,t]) - used
		global total -= used
	end
	println("Correcct shortage: ", shortage)
end
