module ForecastData

using CSV
using Dates
using DataFrames

basepath = normpath(@__DIR__, "../../")

export forecast

function forecast(locations::Array, start_date::Date, end_date::Date;
		level::Symbol=:state, source::Symbol=:ihme,
		forecast_type::Symbol=:active, patient_type::Symbol=:regular, bound_type::Symbol=:mean,
		hospitalized_days::Int=-1, active_from_admitted::Bool=false, merge_gt::Bool=false,
)
	if source == :ihme
		@assert level == :state
		forecast = load_ihme()
		forecast = ihme_filter!(forecast, locations, start_date, end_date)
		patients = ihme_forecast(forecast, forecast_type=forecast_type, patient_type=patient_type, bound_type=bound_type)
	elseif source == :columbia && level == :state
		@assert patient_type == :regular
		forecast = load_columbia(:state)
		if (forecast_type == :active)
			@assert hospitalized_days > 0
			forecast = columbia_filter!(forecast, locations, Date(2020,4,26), Date(2020,6,6))
			columbia_estimate_active!(forecast, hospitalized_days)
		end
		forecast = columbia_filter!(forecast, locations, start_date, end_date)
		patients = columbia_forecast(forecast, forecast_type=forecast_type, bound_type=bound_type)
	elseif source == :columbia && level == :county
		@assert patient_type == :regular
		forecast = load_columbia(:county)
		if (forecast_type == :active)
			@assert hospitalized_days > 0
			forecast = columbia_filter!(forecast, locations, Date(2020,4,26), Date(2020,6,6))
			columbia_estimate_active!(forecast, hospitalized_days)
		end
		forecast = columbia_filter!(forecast, locations, start_date, end_date)
		patients = columbia_forecast(forecast, forecast_type=forecast_type, bound_type=bound_type)
	elseif source == :mit
		@assert level == :state
		@assert forecast_type == :active
		@assert patient_type  == :regular
		@assert bound_type    == :mean
		forecast = load_mit()
		forecast = mit_filter!(forecast, locations, start_date, end_date)
		patients = mit_forecast(forecast)
	elseif source == :cdc_gt || (source == :gt && forecast_type == :active)
		@assert level == :state
		@assert forecast_type == :active
		@assert patient_type  == :regular
		gt = load_cdc_gt()
		gt = cdc_gt_filter!(gt, locations, start_date, end_date)
		patients = cdc_gt_history(gt, bound_type=bound_type)
	elseif source == :covidtracking || (source == :gt && forecast_type == :admitted)
		@assert level == :state
		@assert patient_type  == :regular
		@assert bound_type    == :mean
		forecast = load_covidtracking()
		forecast = covidtracking_filter!(forecast, locations, start_date, end_date)
		patients = covidtracking_history(forecast, forecast_type=forecast_type)
	else
		error("Invalid parameters to forecast.")
	end
end

########################
######## IHME ##########
########################

function load_ihme(;forecast_date::String="2020-06-03")
	forecast_data = CSV.read(joinpath(basepath, "data/forecasts/ihme/$(forecast_date)/forecast.csv"), copycols=true)

	state_list = CSV.read(joinpath(basepath, "data/geography/states.csv"), copycols=true)

	ga = state_list[(state_list.abbrev .== "GA"),:]
	ga.state .= "Georgia_two"
	append!(state_list, ga)

	filter!(row -> row.location_name in state_list.state, forecast_data)

	state_cvt = Dict(state.state => state.abbrev for state in eachrow(state_list))
	forecast_data.state = [state_cvt[row.location_name] for row in eachrow(forecast_data)]

	sort!(forecast_data, [:state, :date])
	return forecast_data
end

function ihme_filter!(forecast::DataFrame, states::Array{String,1}, start_date::Date, end_date::Date)
	@assert states == sort(states)
	@assert start_date <= end_date
	filter!(row -> row.state in states, forecast)
	filter!(row -> start_date <= row.date <= end_date, forecast)
	return forecast
end

function ihme_forecast(forecast::DataFrame; forecast_type::Symbol=:active, patient_type::Symbol=:regular, bound_type::Symbol=:mean)
	forecast[!,:admis_regular_mean] = 0.9 .* forecast.admis_mean
	forecast[!,:allbed_regular_mean] = forecast.allbed_mean - forecast.ICUbed_mean

	col_select = Dict(
		(:admitted, :all, :mean) => :admis_mean,
		(:admitted, :all, :lb)   => :admis_lower,
		(:admitted, :all, :ub)   => :admis_upper,
		(:active, :all, :mean) => :allbed_mean,
		(:active, :all, :lb)   => :allbed_lower,
		(:active, :all, :ub)   => :allbed_upper,
		(:admitted, :icu, :mean) => :newICU_mean,
		(:admitted, :icu, :lb)   => :newICU_lower,
		(:admitted, :icu, :ub)   => :newICU_upper,
		(:active, :icu, :mean) => :ICUbed_mean,
		(:active, :icu, :lb)   => :ICUbed_lower,
		(:active, :icu, :ub)   => :ICUbed_upper,
		(:admitted, :regular, :mean) => :admis_regular_mean,
		(:active, :regular, :mean) => :allbed_regular_mean,
	)
	col = col_select[(forecast_type, patient_type, bound_type)]

	forecast_by_state = groupby(forecast, :state, sort=true)
	patients = vcat([f[:,col]' for f in forecast_by_state]...)
	return Float32.(patients)
end

########################
###### Columbia ########
########################

function load_columbia(level::Symbol=:state; intervention::String="80contact", reopening::String="w5p")
	if level == :state

		folders = readdir(joinpath(basepath, "data/forecasts/columbia/"), join=true)
		fns_a = [joinpath(p, "state_cdchosp_$(intervention).csv") for p in folders]
		fns_b = [joinpath(p, "state_cdchosp_$(intervention)$(reopening).csv") for p in folders]
		fns = vcat(fns_a, fns_b)
		filter!(isfile, fns)

		startdates = [Date(splitpath(p)[end-1]) for p in fns]
		forecasts = []
		for p in fns
			d = Date(splitpath(p)[end-1])
			future_dates = filter(x -> x > d, startdates)
			fd = isempty(future_dates) ? Date(2100) : minimum(future_dates)

			data = CSV.read(p, copycols=true, dateformat="mm/dd/yy")
			data.Date = map(d -> d < Date(2000) ? d + Year(2000) : d, data.Date)
			filter!(r -> r.Date < fd, data)

			push!(forecasts, data)
		end
		forecast_data = vcat(forecasts...)

		state_list = CSV.read(joinpath(basepath, "data/geography/states.csv"), copycols=true)
		filter!(row -> row.location in state_list.state, forecast_data)
		state_cvt = Dict(state.state => state.abbrev for state in eachrow(state_list))
		forecast_data.loc = [state_cvt[s] for s in forecast_data.location]

	else
		forecast_data = CSV.read(joinpath(basepath, "data/forecasts/columbia/2020-04-26/cdchosp_$(intervention).csv"), copycols=true, dateformat="mm/dd/yy")
		forecast_data.loc = forecast_data.fips
		forecast_data.Date = map(d -> d + Year(2000), forecast_data.Date)
	end

	rename!(forecast_data, map(x -> strip(string(x)), names(forecast_data)))
	sort!(forecast_data, [:loc, :Date])
	return forecast_data
end

function columbia_filter!(forecast_data::DataFrame, locations::Array, start_date::Date, end_date::Date)
	@assert locations == sort(locations)
	@assert Date(2020, 4, 12) <= start_date <= end_date <= Date(2020, 7, 3)
	forecast_data = filter(x -> x.loc in locations, forecast_data)
	forecast_data = filter(x -> start_date <= x.Date <= end_date, forecast_data)
	return forecast_data
end

function columbia_estimate_active!(forecast_data::DataFrame, hospitalized_days::Int)
	forecast_data.hosp_active_50 = zeros(Float32, size(forecast_data,1))
	forecast_data.hosp_active_5  = zeros(Float32, size(forecast_data,1))
	forecast_data.hosp_active_95 = zeros(Float32, size(forecast_data,1))
	for (i, s) in enumerate(unique(forecast_data.loc))
		ind = forecast_data.loc .== s
		rows = @view forecast_data[ind,:]
		n = size(rows,1)
		forecast_data[ind, :hosp_active_50] = [sum(rows.hosp_new_50[max(1,i-hospitalized_days):i]) for i in 1:n]
		forecast_data[ind, :hosp_active_5]  = [sum(rows.hosp_new_5[max(1,i-hospitalized_days):i])  for i in 1:n]
		forecast_data[ind, :hosp_active_95] = [sum(rows.hosp_new_95[max(1,i-hospitalized_days):i]) for i in 1:n]
	end
	return forecast_data
end

function columbia_forecast(forecast::DataFrame; forecast_type::Symbol=:active, patient_type::Symbol=:regular, bound_type::Symbol=:mean)

	col_select = Dict(
		(:admitted, :regular, :mean) => :hosp_new_50,
		(:admitted, :regular, :lb)   => :hosp_new_5,
		(:admitted, :regular, :ub)   => :hosp_new_95,
		(:active, :regular, :mean) => :hosp_active_50,
		(:active, :regular, :lb)   => :hosp_active_5,
		(:active, :regular, :ub)   => :hosp_active_95,
	)
	col = col_select[(forecast_type, patient_type, bound_type)]

	forecast_local = groupby(forecast, :loc, sort=true)
	patients = vcat([f[:,col]' for f in forecast_local]...)
	return Float32.(patients)
end

########################
######### MIT ##########
########################

function load_mit(;forecast_date::String="2020-05-15")
	forecast_data = CSV.read(joinpath(basepath, "data/forecasts/mit/$(forecast_date)/forecast.csv"), copycols=true)
	filter!(row -> row.Country == "US", forecast_data)

	state_list = CSV.read(joinpath(basepath, "data/geography/states.csv"), copycols=true)
	state_cvt = Dict(state.state => state.abbrev for state in eachrow(state_list))
	forecast_data.state = map(s -> if haskey(state_cvt, s) state_cvt[s] else "" end, forecast_data.Province)

	return forecast_data
end

function mit_filter!(forecast_data::DataFrame, locations::Array{String,1}, start_date::Date, end_date::Date)
	@assert locations == sort(locations)
	@assert Date(2020,5,15) <= start_date <= end_date <= Date(2020,6,15)

	filter!(row -> start_date <= row.Day <= end_date, forecast_data)
	filter!(row -> row.state in locations, forecast_data)
	return forecast_data
end

function mit_forecast(forecast_data::DataFrame)
	col = Symbol("Active Hospitalized")
	forecast_by_state = groupby(forecast_data, :state, sort=true)
	forecast = vcat([f[:,col]' for f in forecast_by_state]...)
	return Float32.(forecast)
end

########################
####### General ########
########################

function merge_forecast_covidtracking(forecast::Array{Float32,2}, gt::Array{Union{Missing,Float32},2})
	N, T = size(forecast)
	f = copy(forecast)
	g = hcat(gt, fill(missing, N, T-size(gt,2)))
	ind = .~ismissing.(g)
	f[ind] .= g[ind]
	return f
end

end;
