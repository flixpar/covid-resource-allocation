module PatientBlockResults

using DataFrames
using LinearAlgebra
using Dates
using Distributions
using Memoize


function results_all(
		sent::Array{<:Real,4},
		beds::Array{<:Real,2},
		initial::Array{<:Real,2},
		discharged::Array{<:Real,3},
		admitted::Array{<:Real,3},
		los::Array{<:Any,1},
		adj_matrix::BitArray{2},
		transfer_graph::BitArray{2},
		bed_types::Array{Int,1},
		locations::Array{String,1},
		start_date::Date,
)
	B = size(beds,1)
	G, N, T = size(admitted)

	active = results_active_patients(sent, beds, initial, discharged, admitted, los, adj_matrix, transfer_graph, bed_types)

	summary = results_summary(active, sent, locations)
	complete = results_complete(active, sent, admitted, beds, locations, start_date)

	sent_matrix_table = results_sentmatrix_table(sent, locations)
	sent_matrix_vis = results_sentmatrix_vis(sent, initial, admitted, locations)

	sent_to = Dict(locations[i] => locations[row] for (i,row) in enumerate(eachrow(sum(sent, dims=[1,4])[1,:,:,1] .> 0)))

	return (
		summary_table=summary,
		complete_table=complete,
		sent_matrix_table=sent_matrix_table,
		sent_matrix_vis=sent_matrix_vis,
		sent_to=sent_to,
		total_overflow=sum(active.overflow_total),
		total_overflow_nosent=sum(active.overflow_null_total),
		active...,
	)
end

function results_summary(
		active,
		sent::Array{<:Real,4},
		locations::Array{String,1},
)
	summary = DataFrame(
		location=locations,
		total_sent=sum(sent, dims=[1,3,4])[:],
		total_received=sum(sent, dims=[1,2,4])[:],
		overflow=sum(active.overflow_total, dims=2)[:],
	)
	return summary
end

function results_complete(
		active,
		sent::Array{<:Real,4},
		admitted::Array{<:Real,3},
		beds::Array{<:Real,2},
		locations::Array{String,1},
		start_date::Date,
)
	G, N, _, T = size(sent)
	B = size(beds, 1)
	sent_total = sum(sent, dims=1)[1,:,:,:]

	outcomes = DataFrame()
	for (i,s) in enumerate(locations)
		single_state_outcome = DataFrame(
			location=fill(s, T),
			date=start_date .+ Day.(0:T-1),

			sent_total=sum(sent_total[i,:,:], dims=1)[:],
			received_total=sum(sent_total[:,i,:], dims=1)[:],
			new_patients_total=sum(admitted[:,i,:], dims=1)[:],
			capacity_total=fill(sum(beds[:,i]), T),
			active_patients_total=sum(active.active_bybedtype[:,i,:], dims=1)[:],
			active_patients_nosent_total=sum(active.active_null_bybedtype[:,i,:], dims=1)[:],
			overflow_total=sum(active.overflow_bybedtype[:,i,:], dims=1)[:],
			overflow_nosent_total=sum(active.overflow_null_bybedtype[:,i,:], dims=1)[:],
			load_total=active.load_total[i,:],
			load_nosent_total=active.load_null_total[i,:],

			sent_icu=sum(sent[1,i,:,:], dims=1)[:],
			received_icu=sum(sent[1,:,i,:], dims=1)[:],
			new_patients_icu=admitted[1,i,:],
			capacity_icu=fill(beds[1,i], T),
			active_patients_icu=active.active_bybedtype[1,i,:],
			active_patients_nosent_icu=active.active_null_bybedtype[1,i,:],
			overflow_icu=active.overflow_bybedtype[1,i,:],
			overflow_nosent_icu=active.overflow_null_bybedtype[1,i,:],
			load_icu=active.load_bybedtype[1,i,:],
			load_nosent_icu=active.load_null_bybedtype[1,i,:],

			sent_regular=sum(sent[2,i,:,:], dims=1)[:],
			received_regular=sum(sent[2,:,i,:], dims=1)[:],
			new_patients_regular=admitted[1,i,:],
			capacity_regular=fill(beds[2,i], T),
			active_patients_regular=active.active_bybedtype[2,i,:],
			active_patients_nosent_regular=active.active_null_bybedtype[2,i,:],
			overflow_regular=active.overflow_bybedtype[2,i,:],
			overflow_nosent_regular=active.overflow_null_bybedtype[2,i,:],
			load_regular=active.load_bybedtype[2,i,:],
			load_nosent_regular=active.load_null_bybedtype[2,i,:],

			sent_to=[collect(zip(locations[sent_total[i,:,t] .> 0], sent_total[i,sent_total[i,:,t].>0,t])) for t in 1:T],
			sent_from=[collect(zip(locations[sent_total[:,i,t] .> 0], sent_total[sent_total[:,i,t].>0,i,t])) for t in 1:T],
		)
		outcomes = vcat(outcomes, single_state_outcome)
	end

	return outcomes
end

function results_sentmatrix_table(sent::Array{<:Real,4}, locations::Array{String,1})
	sent_matrix = DataFrame(sum(sent, dims=[1,4])[1,:,:,1])
	rename!(sent_matrix, Symbol.(locations))
	insertcols!(sent_matrix, 1, :state => locations)
	return sent_matrix
end

function results_sentmatrix_vis(
		sent::Array{<:Real,4},
		initial_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,3},
		locations::Array{String,1}
)
	sent_total = sum(sent, dims=1)[1,:,:,:]
	selfedges = sum(initial_patients, dims=1)[:] + sum(admitted_patients, dims=[1,3])[:] - sum(sent_total, dims=[2,3])[:]
	sent_vis_matrix = sum(sent_total, dims=3)[:,:,1] + diagm(selfedges)
	sent_vis_matrix = DataFrame(sent_vis_matrix)
	rename!(sent_vis_matrix, Symbol.(locations))
	return sent_vis_matrix
end

function results_active_patients(
		sent::Array{<:Real,4},
		beds::Array{<:Real,2},
		initial::Array{<:Real,2},
		discharged::Array{<:Real,3},
		admitted::Array{<:Real,3},
		los::Array{<:Any,1},
		adj_matrix::BitArray{2},
		transfer_graph::BitArray{2},
		bed_types::Array{Int,1},
)
	B = size(beds,1)
	G, N, T = size(admitted)

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

	groups_bybedtype = [sort(findall(x -> x == b, bed_types)) for b in 1:B]

	@memoize q(g,i,t) = (
		admitted[g,i,t] + tr(g,i,t) + sum(sent[g,j,i,t] - sent[g,i,j,t] for j in 1:N)
	)
	@memoize l(g,i,t) = (
		discharged[g,i,t] + sum(f[g,t-t₁+1] * q(g,i,t₁) for t₁ in 1:t)
	)
	@memoize tr(g,i,t) = (
		any(transfer_graph[:,g]) ?
			sum(l(g₁,i,t) for g₁ in findall(transfer_graph[:,g]))
			: 0
	)
	active_bygroup = [(
		initial[g,i] + sum(sent[g,i,:,t]) + sum((q(g,i,t₁) - l(g,i,t₁)) for t₁ in 1:t)
	) for g in 1:G, i in 1:N, t in 1:T]

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
	active_null_bygroup = [(
		initial[g,i] + sum((q_null(g,i,t₁) - l_null(g,i,t₁)) for t₁ in 1:t)
	) for g in 1:G, i in 1:N, t in 1:T]

	active_bybedtype = [sum(active_bygroup[g,i,t] for g in groups_bybedtype[b]) for b in 1:B, i in 1:N, t in 1:T]
	active_null_bybedtype = [sum(active_null_bygroup[g,i,t] for g in groups_bybedtype[b]) for b in 1:B, i in 1:N, t in 1:T]

	active_total = [sum(active_bygroup[:,i,t]) for i in 1:N, t in 1:T]
	active_null_total = [sum(active_null_bygroup[:,i,t]) for i in 1:N, t in 1:T]

	overflow_bybedtype = [max(0, active_bybedtype[b,i,t] - beds[b,i]) for b in 1:B, i in 1:N, t in 1:T]
	overflow_total = [sum(overflow_bybedtype[:,i,t]) for i in 1:N, t in 1:T]
	load_bybedtype = [active_bybedtype[b,i,t] / beds[b,i] for b in 1:B, i in 1:N, t in 1:T]
	load_total = [sum(active_bybedtype[:,i,t]) / sum(beds[:,i]) for i in 1:N, t in 1:T]

	overflow_null_bybedtype = [max(0, active_null_bybedtype[b,i,t] - beds[b,i]) for b in 1:B, i in 1:N, t in 1:T]
	overflow_null_total = [sum(overflow_null_bybedtype[:,i,t]) for i in 1:N, t in 1:T]
	load_null_bybedtype = [active_null_bybedtype[b,i,t] / beds[b,i] for b in 1:B, i in 1:N, t in 1:T]
	load_null_total = [sum(active_null_bybedtype[:,i,t]) / sum(beds[:,i]) for i in 1:N, t in 1:T]

	return (
		active_bygroup          = active_bygroup,
		active_null_bygroup     = active_null_bygroup,
		active_bybedtype        = active_bybedtype,
		active_null_bybedtype   = active_null_bybedtype,
		active_total            = active_total,
		active_null_total       = active_null_total,
		overflow_bybedtype      = overflow_bybedtype,
		overflow_null_bybedtype = overflow_null_bybedtype,
		overflow_total          = overflow_total,
		overflow_null_total     = overflow_null_total,
		load_bybedtype          = load_bybedtype,
		load_null_bybedtype     = load_null_bybedtype,
		load_total              = load_total,
		load_null_total         = load_null_total,
	)
end

end
