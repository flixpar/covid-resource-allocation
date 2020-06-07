module PatientAllocationResults

using DataFrames
using LinearAlgebra
using Dates


function results_all(
		sent::Array{Float64,3},
		beds::Array{Float32,1},
		initial_patients::Array{Float32,1},
		admitted_patients::Array{Float32,2},
		discharged_patients::Array{Float32,2},
		locations::Array{String,1},
		start_date::Date,
		hospitalized_days::Int,
)
	N, _, T = size(sent)

	summary = results_summary(sent, beds, initial_patients, admitted_patients, discharged_patients, locations, start_date, hospitalized_days)
	complete = results_complete(sent, beds, initial_patients, admitted_patients, discharged_patients, locations, start_date, hospitalized_days)

	sent_matrix_table = results_sentmatrix_table(sent, locations)
	sent_matrix_vis = results_sentmatrix_vis(sent, initial_patients, admitted_patients, locations)

	total_overflow = sum(summary.overflow)
	avgload = sum(summary.average_load)/N

	sent_to = Dict(locations[i] => locations[row] for (i,row) in enumerate(eachrow(sum(sent, dims=3)[:,:,1] .> 0)))

	return (
		total_overflow=total_overflow,
		average_load=avgload,
		summary_table=summary,
		complete_table=complete,
		sent_matrix_table=sent_matrix_table,
		sent_matrix_vis=sent_matrix_vis,
		sent_to=sent_to,
	)
end

function results_summary(
		sent::Array{Float64,3},
		beds::Array{Float32,1},
		initial_patients::Array{Float32,1},
		admitted_patients::Array{Float32,2},
		discharged_patients::Array{Float32,2},
		locations::Array{String,1},
		start_date::Date,
		hospitalized_days::Int,
)
	N, _, T = size(sent)

	active_patients(i,t) = max(0,
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(admitted_patients[i,max(1,t-hospitalized_days):t])
		- sum(sent[i,:,max(1,t-hospitalized_days):t])
		+ sum(sent[:,i,max(1,t-hospitalized_days):t])
	)

	overflow(i,t) = max(0, active_patients(i,t) + sum(sent[i,:,t]) - beds[i])
	overflow(i) = sum(overflow(i,t) for t in 1:T)
	overflow() = sum(overflow(i) for i in 1:N)

	load(i,t) = active_patients(i,t) / beds[i]
	avgload(i) = sum(load(i,t) for t in 1:T)/T
	avgload() = sum(avgload(i) for i in 1:N)/N

	summary = DataFrame(
		state=locations,
		total_sent=sum(sent, dims=[2,3])[:],
		total_received=sum(sent, dims=[1,3])[:],
		overflow=overflow.(1:N),
		average_load=avgload.(1:N),
	)

	return summary
end

function results_complete(
		sent::Array{Float64,3},
		beds::Array{Float32,1},
		initial_patients::Array{Float32,1},
		admitted_patients::Array{Float32,2},
		discharged_patients::Array{Float32,2},
		locations::Array{String,1},
		start_date::Date,
		hospitalized_days::Int,
)
	N, _, T = size(sent)

	active_patients(i,t) = max(0,
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(admitted_patients[i,max(1,t-hospitalized_days):t])
		- sum(sent[i,:,max(1,t-hospitalized_days):t])
		+ sum(sent[:,i,max(1,t-hospitalized_days):t])
	)

	overflow(i,t) = max(0, active_patients(i,t) + sum(sent[i,:,t]) - beds[i])
	overflow(i) = sum(overflow(i,t) for t in 1:T)
	overflow() = sum(overflow(i) for i in 1:N)

	load(i,t) = active_patients(i,t) / beds[i]
	avgload(i) = sum(load(i,t) for t in 1:T)/T
	avgload() = sum(avgload(i) for i in 1:N)/N

	outcomes = DataFrame()
	for (i,s) in enumerate(locations)
		single_state_outcome = DataFrame(
			state=fill(s, T),
			day=start_date .+ Dates.Day.(0:T-1),
			sent=sum(sent[i,:,:], dims=1)[:],
			received=sum(sent[:,i,:], dims=1)[:],
			new_patients=admitted_patients[i,:],
			active_patients=[active_patients(i,t) for t in 1:T],
			capacity=fill(beds[i], T),
			overflow=[overflow(i,t) for t in 1:T],
			load=[load(i,t) for t in 1:T],
			sent_to=[sum(sent[i,:,t])>0 ? collect(zip(locations[sent[i,:,t] .> 0], sent[i,sent[i,:,t].>0,t])) : "[]" for t in 1:T],
			sent_from=[sum(sent[:,i,t])>0 ? collect(zip(locations[sent[:,i,t] .> 0], sent[sent[:,i,t].>0,i,t])) : "[]" for t in 1:T],
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

function results_sentmatrix_vis(sent::Array{Float64,3}, initial_patients::Array{Float32,1},
		admitted_patients::Array{Float32,2}, locations::Array{String,1})
	selfedges = initial_patients + sum(admitted_patients, dims=2)[:] - sum(sent, dims=[2,3])[:]
	sent_vis_matrix = sum(sent, dims=3)[:,:,1] + diagm(selfedges)
	sent_vis_matrix = DataFrame(sent_vis_matrix)
	rename!(sent_vis_matrix, Symbol.(locations))
	return sent_vis_matrix
end

end
