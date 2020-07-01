module LocalData

using Dates
using CSV, JSON
using DataFrames

basepath = normpath(@__DIR__, "../../")

export alldata_nj, load_nj


########################
###### New Jersey ######
########################

function load_nj(start_date::Date, end_date::Date; data_type::Symbol, los::Int=-1, pct_beds_available::Real=1.0)
	data_raw = _load_nj()
	data = filter_nj(data_raw, start_date, end_date)
	if data_type == :active_patients
		return active_nj(data)
	elseif data_type == :admitted_patients
		return admitted_nj(data)
	elseif data_type == :initial_patients
		return initial_nj(start_date, end_date)
	elseif data_type == :discharged_patients
		_initial = initial_nj(start_date, end_date)
		return discharged_nj(_initial, start_date, end_date, los=los)
	elseif data_type == :beds
		return beds_nj(data, pct_beds_available=pct_beds_available)
	elseif data_type == :hospital_names
		return hospitals_nj(data)
	elseif data_type == :hospital_locations
		return locations_nj(data)
	elseif data_type == :all
		return alldata_nj(start_date, end_date, los=los, pct_beds_available=pct_beds_available)
	else
		error("Invalid data_type for load_nj")
	end
end

function _load_nj()
	fn = joinpath(basepath, "data/hospitalizations/newjersey/nj_hosp_2020_06_19.csv")
	data_raw = DataFrame(CSV.File(fn))
	return data_raw
end

function filter_nj(data_raw, start_date::Date, end_date::Date)
	data = filter(row -> start_date <= row.date <= end_date, data_raw)

	_locations = sort(unique(data.hosp_name))
	_N = length(_locations)
	_T = (end_date - start_date).value + 1

	# fill missing days
	for l in _locations
		rows = data[data.hosp_name .== l,:]
		if size(rows,1) != _T
			missing_dates = setdiff(start_date:Day(1):end_date, rows.date)
			for d in missing_dates
				push!(data, Dict(:hosp_id => rows.hosp_id[1], :hosp_name => l, :date => d), cols=:subset)
			end
		end
	end

	# filter out hospitals that never have data
	bed_cols = [
		:bed_avail_criticalcare, :bed_avail_intensivecare, :bed_avail_surgical, :bed_avail_other,
		:bed_inuse_criticalcare, :bed_inuse_intensivecare, :bed_inuse_surgical, :bed_inuse_other,
	]
	data.total_beds = sum(hcat([replace(data[:,b], missing => 0) for b in bed_cols]...), dims=2)[:]
	_beds = maximum(vcat([d[:,:total_beds]' for d in groupby(data, :hosp_name, sort=true)]...), dims=2)[:]
	filter_mask = _beds .!= 0
	locations   = _locations[filter_mask]
	filter!(row -> row.hosp_name in locations, data)

	sort!(data, [:hosp_name, :date])
	return data
end

function admitted_nj(data)
	data_by_hosp = groupby(data, :hosp_name)
	_admitted_patients = vcat([d[:,:admitted]' for d in data_by_hosp]...)
	_admitted_patients = Int.(replace(_admitted_patients, missing => 0))
	return _admitted_patients
end

function active_nj(data)
	data_by_hosp = groupby(data, :hosp_name)
	_active_confirmed = replace(vcat([d[:,:active_confirmed]' for d in data_by_hosp]...), missing => 0)
	_active_suspected = replace(vcat([d[:,:active_suspected]' for d in data_by_hosp]...), missing => 0)
	_active = _active_confirmed + _active_suspected
	return _active
end

function initial_nj(start_date::Date, end_date::Date)
	data_raw = _load_nj()
	data = filter_nj(data_raw, start_date-Day(1), end_date)

	day0_df = filter(row -> row.date == start_date-Day(1), data)
	sort!(day0_df, :hosp_name)

	_initial_confirmed = replace(day0_df.active_confirmed, missing => 0)
	_initial_suspected = replace(day0_df.active_suspected, missing => 0)
	_initial = _initial_confirmed + _initial_suspected

	return _initial
end

function discharged_nj(initial::Array{<:Real,1}, start_date::Date, end_date::Date; los::Int)
	N = length(initial)
	T = (end_date - start_date).value
	_discharged_amt = max.(1, round.(Int, initial / los))
	_discharged = zeros(Int, N, T)
	for i in 1:_N
		for t in 1:_T
			_discharged[i,t] = min(_discharged_amt[i], initial[i] - sum(_discharged[i,:]))
		end
	end
	return _discharged
end

function beds_nj(data; pct_beds_available::Real=1.0)
	bed_cols = [
		:bed_avail_criticalcare, :bed_avail_intensivecare, :bed_avail_surgical, :bed_avail_other,
		:bed_inuse_criticalcare, :bed_inuse_intensivecare, :bed_inuse_surgical, :bed_inuse_other,
	]
	data.total_beds = sum(hcat([replace(data[:,b], missing => 0) for b in bed_cols]...), dims=2)[:]

	data_by_hosp = groupby(data, :hosp_name)
	_beds = maximum(vcat([d[:,:total_beds]' for d in data_by_hosp]...), dims=2)[:]
	beds = _beds .* pct_beds_available

	return beds
end

function hospitals_nj(data)
	return sort(unique(data.hosp_name))
end

function locations_nj(data)
	fn = joinpath(basepath, "data/hospitalizations/newjersey/nj_hosp_locs.csv")
	open(fn, "w") do f
		locs = JSON.parsefile(f)
	end

	hosp_names = sort(unique(data.hosp_name))
	hosp_locs = [locs[n] for n in hosp_names]

	return hosp_locs
end

function alldata_nj(start_date::Date, end_date::Date; pct_beds_available::Real=1.0, los::Int)
	@assert Date(2020, 4, 5) <= start_date <= end_date <= Date(2020, 6, 19)
	@assert los > 0

	fn = joinpath(basepath, "data/hospitalizations/newjersey/nj_hosp_2020_06_19.csv")
	data_raw = DataFrame(CSV.File(fn))
	data = filter(row -> start_date <= row.date <= end_date, data_raw)

	_locations = sort(unique(data.hosp_name))

	_N = length(_locations)
	_T = (end_date - start_date).value + 1

	for l in _locations
		rows = data[data.hosp_name .== l,:]
		if size(rows,1) != _T
			missing_dates = setdiff(start_date:Day(1):end_date, rows.date)
			for d in missing_dates
				push!(data, Dict(:hosp_id => rows.hosp_id[1], :hosp_name => l, :date => d), cols=:subset)
			end
		end
	end
	sort!(data, [:date, :hosp_name])

	data_by_hosp = groupby(data, :hosp_name)
	_admitted_patients = vcat([d[:,:admitted]' for d in data_by_hosp]...)
	_admitted_patients = Int.(replace(_admitted_patients, missing => 0))

	_active_confirmed = replace(vcat([d[:,:active_confirmed]' for d in data_by_hosp]...), missing => 0)
	_active_suspected = replace(vcat([d[:,:active_suspected]' for d in data_by_hosp]...), missing => 0)
	_active = _active_confirmed + _active_suspected

	day0_df = filter(row -> row.date == start_date-Day(1), data_raw)
	for l in _locations
		if l ∉ day0_df.hosp_name
			push!(day0_df, Dict(:hosp_name => l, :date => start_date-Day(1)), cols=:subset)
		end
	end
	sort!(day0_df, :hosp_name)
	_initial_confirmed = replace(day0_df.active_confirmed, missing => 0)
	_initial_suspected = replace(day0_df.active_suspected, missing => 0)
	_initial = _initial_confirmed + _initial_suspected

	_discharged_amt = max.(1, round.(Int, _initial / los))
	_discharged = zeros(Int, _N, _T)
	for i in 1:_N
		for t in 1:_T
			_discharged[i,t] = min(_discharged_amt[i], _initial[i] - sum(_discharged[i,:]))
		end
	end

	bed_cols = [
		:bed_avail_criticalcare, :bed_avail_intensivecare, :bed_avail_surgical, :bed_avail_other,
		:bed_inuse_criticalcare, :bed_inuse_intensivecare, :bed_inuse_surgical, :bed_inuse_other,
	]
	data.total_beds = sum(hcat([replace(data[:,b], missing => 0) for b in bed_cols]...), dims=2)[:]

	data_by_hosp = groupby(data, :hosp_name)
	_beds = maximum(vcat([d[:,:total_beds]' for d in data_by_hosp]...), dims=2)[:]

	filter_mask         = _beds .!= 0
	locations           = _locations[filter_mask]

	admitted_patients   = Float32.(_admitted_patients[filter_mask,:])
	active_patients     = Float32.(_active[filter_mask,:])
	initial_patients    = Float32.(_initial[filter_mask])
	discharged_patients = Float32.(_discharged[filter_mask,:])
	beds                = Float32.(_beds[filter_mask]) * Float32(pct_beds_available)

	return (
		admitted_patients = admitted_patients,
		active_patients = active_patients,
		initial_patients = initial_patients,
		discharged_patients = discharged_patients,
		beds = beds,
		hospitals = locations,
	)
end

end;
