module GeographicData

using CSV
using Dates
using DataFrames

export compute_adjacencies

basepath = normpath(@__DIR__, "../../")

########################
###### Adjacency #######
########################

let
    global dist_matrix, state_abbrev_list

    # load the city distances data
    dist_df = CSV.read(joinpath(basepath, "data/geography/us_city_travel_times.csv"), copycols=true, type=String, silencewarnings=true)
    dist_data = Matrix(dist_df[:,2:end])

    # load state data
    state_list = CSV.read(joinpath(basepath, "data/geography/state_names.csv"), copycols=true)
    sort!(state_list, :Abbreviation)
    state_abbrev_list = state_list.Abbreviation

    # convert distances to seconds
    function cvt_duration(s::Union{String,Missing})
        if ismissing(s) return 0 end
        parts = parse.(Int, split(s, ":"))
        return 3600*parts[1] + 60*parts[2] + parts[3]
    end
    city_distances = cvt_duration.(dist_data);

    # build list of cities to use
    capital_cities = Dict(state.Abbreviation => state.Capitol * ", " * state.Abbreviation for state in eachrow(state_list))
    alt_cities = Dict(
        "AK" => "Anchorage, AK",
        "DE" => "Wilmington, DE",
        "KY" => "Louisville, KY",
        "ME" => "Portland, ME",
        "MD" => "Baltimore, MD",
        "MO" => "Kansas City, MO",
        "MT" => "Billings, MT",
        "NH" => "Manchester, NH",
        "SD" => "Sioux Falls, SD",
        "VT" => "Manchester, NH",
        "NY" => "New York City, NY",
    );
    capital_cities = merge(capital_cities, alt_cities)

    order = sortperm(collect(keys(capital_cities)))
    selected_city_names = collect(values(capital_cities))[order]

    # get indicies of selected cities
    matrix_cities = String.(names(dist_df)[2:end])
    selected_cities_ind = [findfirst(x -> x == city_name, matrix_cities) for city_name in selected_city_names]

    # compute distance matrix
    dist_matrix = city_distances[selected_cities_ind, selected_cities_ind]

end

function compute_adjacencies(states; hrs::Real=4)
    @assert states == sort(states)
    ind = [findfirst(x -> x == s, state_abbrev_list) for s in states]
    dist = dist_matrix[ind, ind]

    dist_threshold = 3600 * hrs
    adj_matrix = 0 .< dist .<= dist_threshold

    return adj_matrix
end

end;
