module BedsData

using CSV
using Dates
using DataFrames

export compute_beds

basepath = normpath(@__DIR__, "../../")

########################
######## Beds ##########
########################

let
    global beds_data

    # load the beds data
    beds_data = CSV.read(joinpath(basepath, "data/hospitals/hospital_locations.csv"), copycols=true)

    # load state info
    state_list = CSV.read(joinpath(basepath, "data/geography/state_names.csv"), copycols=true)
    state_abbrev_list = sort(state_list.Abbreviation)

    # filter
    filter!(row -> row.BEDS > 0, beds_data)
    filter!(row -> row.STATE in state_abbrev_list, beds_data)
    filter!(row -> row.TYPE == "GENERAL ACUTE CARE", beds_data)

    # aggregate by state
    beds_data = by(beds_data, :STATE, :BEDS => sum)

    # reorder states
    sort!(beds_data, :STATE)

end

function compute_beds(states; pct_beds_available::Real=0.25)
    @assert states == sort(states)
    beds_df = filter(row -> row.STATE in states, beds_data)
    beds = beds_df.BEDS_sum * pct_beds_available
    return Float32.(beds)
end

end;
