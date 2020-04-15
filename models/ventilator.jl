using JuMP
using Gurobi

function ventilator_model_optimize(initial_resource::Array{Int,1}, needs::Array{Int,2}, growth::Array{Int,1})
	N, T = size(needs)

	@assert(size(initial_resource, 1) == N)
	@assert(size(growth, 1) == T)

	model = Model(Gurobi.Optimizer)
	@variable(model, sent[1:N,1:T] >= 0)
	@variable(model, recieved[1:N,1:T] >= 0)

	shortage = 0
	for t = 1:T
		shortage += sum(needs[:,t]) - (sum(initial_resource) + sum(recieved[:,1:t]) - sum(sent[:,1:t]))
	end
	@objective(model, Min, shortage)

	# amount sent by hospital i at time t ≦ amount available at hospital i at time t
	@constraint(model, [t=1:T],
		sent[:,t] .<=
			initial_resource
			+ sum(recieved[:,1:t-1], dims=2)
			- sum(sent[:,1:t-1], dims=2)
	)

	# hospitals don't stockpile more than they need
	@constraint(model, [t=1:T],
		needs[:,t] .>=
			initial_resource
			+ sum(recieved[:,1:t], dims=2)
			- sum(sent[:,1:t], dims=2)
	)

	# total amount recieved ≦ total amount sent plus total amount stockpiled ∀ t
	@constraint(model, [t=1:T], sum(recieved[:,t]) <= sum(sent[:,t]) + sum(growth[1:t]) + sum(sent[:,1:t-1]) - sum(recieved[:,1:t-1]))

	optimize!(model)
	return model
end

if abspath(PROGRAM_FILE) == @__FILE__
	N, T = 20, 60
	initial_resource = rand(1:40, N)
	needs = rand(1:50, N, T)
	growth = rand(0:10, T)

	model = ventilator_model_optimize(initial_resource, needs, growth)

	sent = value.(model[:sent])
	recieved = value.(model[:recieved])

	shortage = sum(
		sum(needs, dims=1)[:] - (
			sum(initial_resource)
			.+ cumsum(sum(recieved, dims=1)[:])
			.- cumsum(sum(sent, dims=1)[:])
		)
	)
	println("Shortage: ", shortage)
	println("Sent: ", sum(sent))
	println("Recieved: ", sum(recieved))

	shortage = sum(max.(0, sum(needs, dims=1)[:] - (sum(initial_resource) .+ cumsum(growth))))
	println("Correct shortage: ", shortage)
end
