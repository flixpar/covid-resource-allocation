module LocalDataCommon

using Distributions
using LinearAlgebra: norm
using ProgressMeter
using BlackBoxOptim

export estimate_admitted_discharged
export estimate_discharged
export estimate_admitted_uncertainty
export build_carepaths_data


function estimate_admitted_discharged(data; use_rounding=false, optim_time=1.0)
	N, T = data.N, data.T

	function unpack_combined_params(i, params)
		_admitted = params[T+1:end]
		_discharged = params[1:T]

		_discharged = _discharged .* (data.initial[i] / sum(_discharged))

		if use_rounding
			_admitted = round.(Int, _admitted)
			_discharged = round.(Int, _discharged)
		end

		return (
			admitted = _admitted,
			discharged = _discharged,
		)
	end

	function combined_score_func(i, params; s_penalty=0.0)
		z = unpack_combined_params(i, params)
		_admitted, _discharged = z.admitted, z.discharged

		_active = compute_active_nosent(data.initial[i], _discharged, _admitted, data.los_dist, use_rounding=use_rounding)
		score = norm(_active - data.active[i,:], 2)
		smooth_score = norm(_active[2:end] - _active[1:end-1], 2)
		return score + (s_penalty * smooth_score)
	end

	function combined_param_bounds(i)
		_admitted_ub = fill(0.2 * maximum(data.active[i,:]), T)
		_discharged_ub = 1 .- cdf.(data.los_dist, 1:T)
		ub = vcat(_discharged_ub, _admitted_ub)
		bds = collect(zip(zeros(Float64, length(ub)), ub))
		return bds
	end

	admitted_sim = zeros(Float64, N, T)
	discharged_sim = zeros(Float64, N, T)
	@showprogress for i in 1:N
		if maximum(data.active[i,:]) == 0.0
			continue
		end

		local_score_func = params -> combined_score_func(i, params, s_penalty=0.0)
		local_param_bounds = combined_param_bounds(i)

		results_combined = bboptimize(
			local_score_func,
			SearchRange = local_param_bounds,
			Method = :adaptive_de_rand_1_bin_radiuslimited,
			TraceMode = :silent,
			MaxTime = optim_time,
			RandomizeRngSeed = false,
			RngSeed = 0,
		)

		_local_params = best_candidate(results_combined)
		local_params = unpack_combined_params(i, _local_params)
		local_admitted, local_discharged = local_params.admitted, local_params.discharged
		admitted_sim[i,:] = local_admitted
		discharged_sim[i,:] = local_discharged
	end

	if use_rounding
		admitted_sim = Int.(admitted_sim)
		discharged_sim = Int.(discharged_sim)
	end

	return (
		admitted = admitted_sim,
		discharged = discharged_sim,
	)
end

function estimate_discharged(data; use_rounding=false, optim_time=1.0)
	N, T = data.N, data.T

	function unpack_params(i, _discharged)
		_discharged = _discharged .* (data.initial[i] / sum(_discharged))
		if use_rounding
			_discharged = round.(Int, _discharged)
		end
		return _discharged
	end

	function score_func(i, params; s_penalty=0.0)
		_discharged = unpack_params(i, params)
		_active = compute_active_nosent(data.initial[i], _discharged, data.admitted[i,:], data.los_dist, use_rounding=use_rounding)
		score = norm(_active - data.active[i,:], 2)
		smooth_score = norm(_active[2:end] - _active[1:end-1], 2)
		return score + (s_penalty * smooth_score)
	end

	function param_bounds(i)
		ub = 1 .- cdf.(data.los_dist, 1:T)
		bds = collect(zip(zeros(Float64, length(ub)), ub))
		return bds
	end

	discharged_sim = zeros(Float64, N, T)
	@showprogress for i in 1:N
		if maximum(data.active[i,:]) == 0.0
			continue
		end

		local_score_func = params ->score_func(i, params, s_penalty=0.0)
		local_param_bounds = param_bounds(i)

		r = bboptimize(
			local_score_func,
			SearchRange = local_param_bounds,
			Method = :adaptive_de_rand_1_bin_radiuslimited,
			TraceMode = :silent,
			MaxTime = optim_time,
			RandomizeRngSeed = false,
			RngSeed = 0,
		)

		_local_params = best_candidate(r)
		discharged_sim[i,:] = unpack_params(i, _local_params)
	end

	if use_rounding
		discharged_sim = Int.(discharged_sim)
	end

	return discharged_sim
end

function compute_active_nosent(_initial::Array{<:Real,1}, _discharged::Array{<:Real,2},
		_admitted::Array{<:Real,2}, los_dist::Distribution; use_rounding=false)
	N, T = size(_admitted)
	_L = 1.0 .- cdf.(los_dist, 0:T)
	_active = [(
		_initial[i]
		- sum(_discharged[i,1:t])
		+ sum(_L[t-t₁+1] * _admitted[i,t₁] for t₁ in 1:t)
	) for i in 1:N, t in 1:T]
	if use_rounding
		_active = round.(Int, _active)
	end
	return _active
end

function compute_active_nosent(_initial::Real, _discharged::Array{<:Real,1},
		_admitted::Array{<:Real,1}, los_dist::Distribution; use_rounding=false)
	T = length(_admitted)
	_L = 1.0 .- cdf.(los_dist, 0:T)
	_active = [(
		_initial
		- sum(_discharged[1:t])
		+ sum(_L[t-t₁+1] * _admitted[t₁] for t₁ in 1:t)
	) for t in 1:T]
	if use_rounding
		_active = round.(Int, _active)
	end
	return _active
end

function estimate_admitted_uncertainty(data, admitted)
	admitted_uncertainty = 0.05 * maximum(admitted, dims=2)[:]
	admitted_uncertainty = repeat(admitted_uncertainty, 1, data.T)
	admitted_uncertainty = min.(admitted_uncertainty, admitted)
	return admitted_uncertainty
end

function roundint(x::Real, direction::Symbol=:nearest)::Int
	rounddir = RoundNearest
	if direction == :down
		rounddir = RoundDown
	elseif direction == :up
		rounddir = RoundUp
	end
	return round(Int, x, rounddir)
end

function build_carepaths_data(data)
	N, T = data.N, data.T

	stack0(A...) = permutedims(cat(A..., dims=ndims(A[1])+1), [ndims(A[1])+1; 1:ndims(A[1])])
	beds_grouped = stack0(data.icu.beds, data.ward.beds)
	admitted_grouped = stack0(zeros(Float64, N, T), data.icu.admitted, zeros(Float64, N, T), data.ward.admitted)
	admitted_uncertainty_grouped = stack0(zeros(Float64, N, T), data.icu.admitted_uncertainty, zeros(Float64, N, T), data.ward.admitted_uncertainty)
	initial_grouped = stack0(data.icu.initial, zeros(Float64, N), zeros(Float64, N), data.ward.initial)
	discharged_grouped = stack0(data.icu.discharged, zeros(Float64, N, T), zeros(Float64, N, T), data.ward.discharged)

	group_transfer_graph = BitArray([
		0 0 1 0
		1 0 0 0
		0 0 0 0
		0 0 0 0
	])

	_p = zeros(Float64, T+1)
	_p[2+1] = 1
	los_dist_toicu = DiscreteNonParametric(0:T, _p)
	_p = zeros(Float64, T+1)
	_p[3+1] = 1
	los_dist_fromicu = DiscreteNonParametric(0:T, _p)
	los_bygroup = [data.icu.los_dist, los_dist_toicu, los_dist_fromicu, data.ward.los_dist]

	carepath_data = merge(data, (
		G = 4,
		B = 2,

		beds = beds_grouped,
		admitted = admitted_grouped,
		admitted_uncertainty = admitted_uncertainty_grouped,
		initial = initial_grouped,
		discharged = discharged_grouped,

		groups = [:icu, :to_icu, :from_icu, :ward],
		bedtypes = [:icu, :ward],
		los_bygroup = los_bygroup,
		bedtype_bygroup = [1, 2, 2, 2],
		group_transfer_graph = group_transfer_graph,
	))
	return carepath_data
end

end;
