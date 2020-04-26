module ForecastData

using CSV
using Dates
using DataFrames

export compute_ihme_forecast, compute_ihme_forecast_net

basepath = normpath(@__DIR__, "../../")

########################
######## IHME ##########
########################

let
    global ihme_forecast

    # load the forecast data
    forecast_data = CSV.read(joinpath(basepath, "data/forecasts/ihme_2020_04_12/forecast.csv"), copycols=true)

    # load state info
    state_list = CSV.read(joinpath(basepath, "data/geography/state_names.csv"), copycols=true)
    sort!(state_list, :Abbreviation)
    state_name_list = state_list.State
    state_abbrev_list = state_list.Abbreviation

    # filter to US states
    filter!(row -> row.location_name in state_name_list, forecast_data)

    # add state abbreviations
    state_cvt = Dict(state.State => state.Abbreviation for state in eachrow(state_list))
    forecast_data.state = [state_cvt[row.location_name] for row in eachrow(forecast_data)]

    # sort
    ihme_forecast = sort(forecast_data, [:state, :date])

    # compute net change
    ihme_forecast.allbed_mean_net = zeros(Float32, size(ihme_forecast,1))
    for (i, s) in enumerate(state_abbrev_list)
        rows = ihme_forecast[ihme_forecast.state .== s,:]
        ihme_forecast[ihme_forecast.state .== s, :allbed_mean_net] = rows.allbed_mean - [rows.allbed_mean[1]; rows.allbed_mean[1:end-1]]
    end

end

function compute_ihme_forecast_net(start_date, end_date, states)
    @assert states == sort(states)
    forecast_local = filter(row -> row.state in states, ihme_forecast)

    patients_start = Float32.(filter(row -> row.date == start_date-Dates.Day(1), forecast_local).allbed_mean)

    filter!(row -> start_date <= row.date <= end_date, forecast_local)
    forecast_by_state = groupby(forecast_local, :state, sort=true)
    patients_net = vcat([f.allbed_mean_net' for f in forecast_by_state]...)
    patients_net = Float32.(patients_net)

    return patients_start, patients_net
end

function compute_ihme_forecast(start_date, end_date, states)
    @assert states == sort(states)

    forecast_local = filter(row -> row.state in states, ihme_forecast)
    filter!(row -> start_date <= row.date <= end_date, forecast_local)

    forecast_by_state = groupby(forecast_local, :state, sort=true)
    patients = vcat([f.allbed_mean' for f in forecast_by_state]...)

    return Float32.(patients)
end

end;
