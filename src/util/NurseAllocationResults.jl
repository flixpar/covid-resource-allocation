module NurseAllocationResults

using DataFrames
using LinearAlgebra
using Dates


function results_all(
		sent::Array{Float64,3},
		initial_nurses::Array{Float32,1},
		demand::Array{Float32,2},
		locations::Array{String,1},
		start_date::Date,
)
	N, _, T = size(sent)

	summary = results_summary(sent, initial_nurses, demand, locations, start_date)
	complete = results_complete(sent, initial_nurses, demand, locations, start_date)

	sent_matrix_table = results_sentmatrix_table(sent, locations)
	sent_matrix_vis = results_sentmatrix_vis(sent, initial_nurses, locations)

	avgload = sum(complete.load)/(N*T)

	sent_to = Dict(locations[i] => locations[row] for (i,row) in enumerate(eachrow(sum(sent, dims=3)[:,:,1] .> 0)))
	netsent = results_netsent(sent::Array{Float64,3}, start_date::Date)

	return (
		summary_table=summary,
		complete_table=complete,
		sent_matrix_table=sent_matrix_table,
		sent_matrix_vis=sent_matrix_vis,
		sent_to=sent_to,
		netsent=netsent,
		average_load=avgload,
	)
end

function results_summary(
		sent::Array{Float64,3},
		initial_nurses::Array{Float32,1},
		demand::Array{Float32,2},
		locations::Array{String,1},
		start_date::Date,
)
	N, _, T = size(sent)

	current_nurses(i,t) = initial_nurses[i] - sum(sent[i,:,1:t]) + sum(sent[:,i,1:t])
	shortage(i,t) = max(0, demand[i,t] - current_nurses(i,t))
	shortage(i) = sum(shortage(i,t) for t=1:T)
	shortage() = sum(shortage(i) for i in 1:N)

	load(i,t) = demand[i,t] / max(1, current_nurses(i,t))
	avgload(i) = sum(load(i,t) for t in 1:T)/T
	avgload() = sum(avgload(i) for i in 1:N)/N

	total_demand = sum(demand, dims=2)[:]

	summary = DataFrame(
		state=locations,
		total_sent=sum(sent, dims=[2,3])[:],
		total_received=sum(sent, dims=[1,3])[:],
		initial_nurses=initial_nurses,
		total_nurse_days=[sum(current_nurses(i,t) for t in 1:T) for i in 1:N],
		total_demand=total_demand,
		total_shortage=shortage.(1:N),
		# average_load=avgload.(1:N),
	)

	return summary
end

function results_complete(
		sent::Array{Float64,3},
		initial_nurses::Array{Float32,1},
		demand::Array{Float32,2},
		locations::Array{String,1},
		start_date::Date,
)
	N, _, T = size(sent)

	current_nurses(i,t) = initial_nurses[i] - sum(sent[i,:,1:t]) + sum(sent[:,i,1:t])
	shortage(i,t) = max(0, demand[i,t] - current_nurses(i,t))
	shortage(i) = sum(shortage(i,t) for t=1:T)
	shortage() = sum(shortage(i) for i in 1:N)

	load(i,t) = demand[i,t] / max(1, current_nurses(i,t))
	avgload(i) = sum(load(i,t) for t in 1:T)/T
	avgload() = sum(avgload(i) for i in 1:N)/N

	outcomes = DataFrame()
	for (i,s) in enumerate(locations)
		single_state_outcome = DataFrame(
			state=fill(s, T),
			date=start_date .+ Dates.Day.(0:T-1),
			sent=sum(sent[i,:,:], dims=1)[:],
			received=sum(sent[:,i,:], dims=1)[:],

			initial_nurses=fill(initial_nurses[i], T),
			current_nurses=current_nurses.(i,1:T),

			demand=demand[i,:],
			shortage=[shortage(i,t) for t in 1:T],
			load=[load(i,t) for t in 1:T],

			# sent_to=[sum(sent[i,:,t])>0 ? collect(zip(locations[sent[i,:,t] .> 0], sent[i,sent[i,:,t].>0,t])) : "[]" for t in 1:T],
			# sent_from=[sum(sent[:,i,t])>0 ? collect(zip(locations[sent[:,i,t] .> 0], sent[sent[:,i,t].>0,i,t])) : "[]" for t in 1:T],
		)
		outcomes = vcat(outcomes, single_state_outcome)
	end

	return outcomes
end

function results_sentmatrix_table(sent::Array{Float64,3}, locations::Array{String,1})
	sent_matrix = DataFrame(sum(sent, dims=3)[:,:,1])
	rename!(sent_matrix, Symbol.(locations))
	insertcols!(sent_matrix, 1, :state => locations)
	return sent_matrix
end

function results_sentmatrix_vis(sent::Array{Float64,3}, initial_nurses::Array{Float32,1}, locations::Array{String,1})
	selfedges = initial_nurses - sum(sent, dims=[2,3])[:]
	sent_vis_matrix = sum(sent, dims=3)[:,:,1] + diagm(selfedges)
	sent_vis_matrix = DataFrame(sent_vis_matrix)
	rename!(sent_vis_matrix, Symbol.(locations))
	return sent_vis_matrix
end

function results_netsent(sent::Array{Float64,3}, start_date::Date)
	N, _, T = size(sent)
	net_sent = sum(sent, dims=2)[:,1,:] .- sum(sent, dims=1)[1,:,:]
	net_sent = DataFrame(Matrix(net_sent))
	rename!(net_sent, Symbol.(start_date .+ Dates.Day.(0:T-1)))
	return net_sent
end

end;
