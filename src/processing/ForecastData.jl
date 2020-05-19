module ForecastData

using CSV
using Dates
using DataFrames

basepath = normpath(@__DIR__, "../../")

export forecast, merge_forecast_gt

function forecast(locations::Array, start_date::Date, end_date::Date;
        level::Symbol=:state, source::Symbol=:ihme,
        forecast_type::Symbol=:active, patient_type::Symbol=:regular, bound_type::Symbol=:mean,
        hospitalized_days::Int=-1,
)
    if level == :state && source == :ihme
        forecast = load_ihme()
        forecast = ihme_filter!(forecast, locations, start_date, end_date)
        patients = ihme_forecast(forecast, forecast_type=forecast_type, patient_type=patient_type, bound_type=bound_type)
        return patients
    elseif level == :state && source == :columbia
        forecast = load_columbia(:state)
        if (forecast_type == :active)
            @assert hospitalized_days > 0
            forecast = columbia_filter!(forecast, locations, Date(2020,4,26), Date(2020,6,6))
            columbia_estimate_active!(forecast, hospitalized_days)
        end
        forecast = columbia_filter!(forecast, locations, start_date, end_date)
        @assert patient_type == :regular
        patients = columbia_forecast(forecast, forecast_type=forecast_type, bound_type=bound_type)
        return patients
    elseif level == :county && source == :columbia
        forecast = load_columbia(:county)
        if (forecast_type == :active)
            @assert hospitalized_days > 0
            forecast = columbia_filter!(forecast, locations, Date(2020,4,26), Date(2020,6,6))
            columbia_estimate_active!(forecast, hospitalized_days)
        end
        forecast = columbia_filter!(forecast, locations, start_date, end_date)
        @assert patient_type == :regular
        patients = columbia_forecast(forecast, forecast_type=forecast_type, bound_type=bound_type)
        return patients
    elseif level == :state && source == :mit
        @assert forecast_type == :active
        @assert patient_type  == :regular
        @assert bound_type    == :mean
        forecast = load_mit()
        forecast = mit_filter!(forecast, locations, start_date, end_date)
        patients = mit_forecast(forecast)
        return patients
    elseif level == :state && (source == :covidtracking || source == :gt)
        @assert forecast_type == :active
        @assert patient_type  == :regular
        @assert bound_type    == :mean
        forecast = load_covidtracking()
        forecast = covidtracking_filter!(forecast, locations, start_date, end_date)
        patients = covidtracking_history(forecast)
        return patients
    else
        error("Invalid parameters to forecast.")
    end
end

########################
######## IHME ##########
########################

function load_ihme(;forecast_date::String="2020-05-10")
    forecast_data = CSV.read(joinpath(basepath, "data/forecasts/ihme/$(forecast_date)/forecast.csv"), copycols=true)

    state_list = CSV.read(joinpath(basepath, "data/geography/states.csv"), copycols=true)
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
    forecast_by_state = groupby(forecast, :state, sort=true)

    col_select = Dict(
        (:admitted, :regular, :mean) => :admis_mean,
        (:admitted, :regular, :lb)   => :admis_lower,
        (:admitted, :regular, :ub)   => :admis_upper,
        (:active, :regular, :mean) => :allbed_mean,
        (:active, :regular, :lb)   => :allbed_lower,
        (:active, :regular, :ub)   => :allbed_upper,
        (:admitted, :icu, :mean) => :newICU_mean,
        (:admitted, :icu, :lb)   => :newICU_lower,
        (:admitted, :icu, :ub)   => :newICU_upper,
        (:active, :icu, :mean) => :ICUbed_mean,
        (:active, :icu, :lb)   => :ICUbed_lower,
        (:active, :icu, :ub)   => :ICUbed_upper,
    )
    col = col_select[(forecast_type, patient_type, bound_type)]

    patients = vcat([f[:,col]' for f in forecast_by_state]...)
    return Float32.(patients)
end

# function ihme_compute_discharged!(ihme_forecast::DataFrame)
#     ihme_forecast.allbed_mean_net = zeros(Float32, size(ihme_forecast,1))
#     for (i, s) in enumerate(state_abbrev_list)
#         rows = ihme_forecast[ihme_forecast.state .== s,:]
#         ihme_forecast[ihme_forecast.state .== s, :allbed_mean_net] = rows.allbed_mean - [rows.allbed_mean[1]; rows.allbed_mean[1:end-1]]
#     end
#     ihme_forecast.discharged_mean = ihme_forecast.allbed_mean_net - ihme_forecast.admis_mean
#     return ihme_forecast
# end

########################
###### Columbia ########
########################

function load_columbia(level::Symbol=:state; forecast_date::String="2020-04-26", intervention::String="60contact")
    if level == :state
        forecast_data = CSV.read(joinpath(basepath, "data/forecasts/columbia/$(forecast_date)/cdc_hosp/state_cdchosp_$(intervention).csv"), copycols=true, dateformat="mm/dd/yy")
        state_list = CSV.read(joinpath(basepath, "data/geography/states.csv"), copycols=true)
        filter!(row -> row.location in state_list.state, forecast_data)
        state_cvt = Dict(state.state => state.abbrev for state in eachrow(state_list))
        forecast_data.loc = [state_cvt[s] for s in forecast_data.location]
    else
        forecast_data = CSV.read(joinpath(basepath, "data/forecasts/columbia/$(forecast_date)/cdc_hosp/cdchosp_$(intervention).csv"), copycols=true, dateformat="mm/dd/yy")
        forecast_data.loc = forecast_data.fips
    end
    rename!(forecast_data, map(x -> strip(string(x)), names(forecast_data)))
    forecast_data.Date = map(d -> d + Dates.Year(2000), forecast_data.Date)
    sort!(forecast_data, [:loc, :Date])
    return forecast_data
end

function columbia_filter!(forecast_data::DataFrame, locations::Array, start_date::Date, end_date::Date)
    @assert locations == sort(locations)
    @assert Date(2020, 4, 26) <= start_date <= end_date <= Date(2020, 6, 6)
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
#### COVIDTracking #####
########################

function load_covidtracking()
    gt = CSV.read(joinpath(basepath, "data/forecasts/covidtracking/2020-05-15/states_daily.csv"), copycols=true, silencewarnings=true)
    gt.date = map(d -> Date(string(d), "yyyymmdd"), gt.date)
    return gt
end

function covidtracking_filter!(gt::DataFrame, locations::Array{String,1}, start_date::Date, end_date::Date)
    @assert locations == sort(locations)
    @assert start_date <= end_date
    @assert end_date <= Date(2020, 5, 15)

    filter!(row -> start_date <= row.date <= end_date, gt)
    filter!(row -> row.state in locations, gt)
    return gt
end

function covidtracking_history(gt::DataFrame; keepmissings::Bool=false)
    groups = groupby(gt, :state, sort=true)
    patients = vcat([g.hospitalizedCurrently' for g in groups]...)
    if !keepmissings
        patients[ismissing.(patients)] .= 0
    end
    return Float32.(patients)
end

########################
####### General ########
########################

function merge_forecast_gt(forecast::Array{Float32,2}, gt::Array{Union{Missing,Float32},2})
    N, T = size(forecast)
    f = copy(forecast)
    g = hcat(gt, fill(missing, N, T-size(gt,2)))
    ind = .~ismissing.(g)
    f[ind] .= g[ind]
    return f
end

end;
