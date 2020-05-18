module BedsData

using CSV
using Dates
using DataFrames

export compute_beds, compute_beds_by_type

basepath = normpath(@__DIR__, "../../")

########################
###### Gov Data ########
########################

let
    global beds_data

    # load the beds data
    beds_data = CSV.read(joinpath(basepath, "data/hospitals/hospital_locations.csv"), copycols=true)

    # load state info
    state_list = CSV.read(joinpath(basepath, "data/geography/states.csv"), copycols=true)
    state_abbrev_list = sort(state_list.abbrev)

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

###########################
## Definitive Healthcare ##
###########################

let
	global beds_data_defininitive

	beds_data_defininitive = CSV.read(joinpath(basepath, "data/hospitals/Definitive_Healthcare__USA_Hospital_Beds.csv"), copycols=true)
	filter!(row -> !(row.HOSPITAL_TYPE in ["Psychiatric Hospital", "Rehabilitation Hospital"]), beds_data_defininitive)
	filter!(row -> !ismissing(row.HQ_STATE), beds_data_defininitive)
end

function compute_beds_by_type(states::Array{String,1}; bed_type::Symbol=:staffed)
    @assert states == sort(states)

	bed_type_cvt = Dict(
		:staffed => :NUM_STAFFED_BEDS,
		:licensed => :NUM_LICENSED_BEDS,
		:icu => :NUM_ICU_BEDS,
		:adult_icu => :ADULT_ICU_BEDS,
	)
	col = bed_type_cvt[bed_type]

	beds_df = filter(row -> row.HQ_STATE in states, beds_data_defininitive)
	filter!(row -> !ismissing(row[col]), beds_df)
	filter!(row -> row[col] > 0, beds_df)

	beds_df = by(beds_df, :HQ_STATE, beds_selected = (col => sum))
	sort!(beds_df, :HQ_STATE)

    return Float32.(beds_df.beds_selected)
end

end;
