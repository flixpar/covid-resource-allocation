module BedsData

using CSV
using Dates
using DataFrames

export n_beds

basepath = normpath(@__DIR__, "../../")

function n_beds(locations::Array; level::Symbol=:state, source::Symbol=:definitivehc, bed_type::Symbol=:all, pct_beds_available::Real=0.25)
	if     level == :state    && source == :definitivehc
		beds_data = load_definitivehc()
		beds_data = definitivehc_filter!(beds_data, locations, level=level)
		beds_data = definitivehc_select_type!(beds_data, bed_type=bed_type)
		beds = definitivehc_aggregate(beds_data)
	elseif level == :state    && source == :gov
		@assert bed_type == :all
		beds_data = load_gov()
		beds = compute_beds_gov(beds_data, locations)
	elseif level == :county   && source == :definitivehc
		beds_data = load_definitivehc()
		beds_data = definitivehc_filter!(beds_data, locations, level=level)
		beds_data = definitivehc_select_type!(beds_data, bed_type=bed_type)
		beds = definitivehc_aggregate(beds_data, locations=locations)
	elseif level == :hospital && source == :definitivehc
		beds_data = load_definitivehc()
		beds_data = definitivehc_filter!(beds_data, locations, level=level)
		beds_data = definitivehc_select_type!(beds_data, bed_type=bed_type)
		sort!(beds_data, :selected_location)
		beds = beds_data.selected_beds
	else
		error("Invalid parameters to compute_beds.")
	end

	return beds * Float32(pct_beds_available)
end

########################
###### data.gov ########
########################

function load_gov()
    beds_data = CSV.read(joinpath(basepath, "data/hospitals/gov.csv"), copycols=true)

    filter!(row -> row.BEDS > 0, beds_data)
    filter!(row -> row.TYPE == "GENERAL ACUTE CARE", beds_data)

    beds_data = combine(groupby(beds_data, :STATE), :BEDS => sum)
    sort!(beds_data, :STATE)

	return beds_data
end

function compute_beds_gov(beds_data::DataFrame, states::Array{String,1})
    @assert states == sort(states)
    beds_df = filter(row -> row.STATE in states, beds_data)
    return Float32.(beds_df.BEDS_sum)
end

###########################
## Definitive Healthcare ##
###########################

function load_definitivehc()
	beds_data = CSV.read(joinpath(basepath, "data/hospitals/definitivehc.csv"), copycols=true)
	filter!(row -> !(row.HOSPITAL_TYPE in ["Psychiatric Hospital", "Rehabilitation Hospital"]), beds_data)
	return beds_data
end

function definitivehc_filter!(beds_data::DataFrame, locations::Array; level::Symbol=:state)
	if level == :state
		filter!(row -> !ismissing(row.HQ_STATE), beds_data)
		beds_data.selected_location = beds_data.HQ_STATE
	elseif level == :county
		filter!(row -> !ismissing(row.FIPS), beds_data)
		beds_data.selected_location = Int.(beds_data.FIPS)
	elseif level == :hospital
		filter!(row -> !ismissing(row.FIPS), beds_data)
		beds_data.selected_location = Int.(beds_data.FIPS)
		return beds_data
	else
		error("Invalid parameters.")
	end
	beds_data = filter(row -> row.selected_location in locations, beds_data)
	return beds_data
end

function definitivehc_select_type!(beds_data::DataFrame; bed_type::Symbol=:all)
	bed_type_cvt = Dict(
		:all       => :NUM_STAFFED_BEDS,
		:staffed   => :NUM_STAFFED_BEDS,
		:licensed  => :NUM_LICENSED_BEDS,
		:icu       => :NUM_ICU_BEDS,
		:adult_icu => :ADULT_ICU_BEDS,
		:regular   => :NUM_NON_ICU_BEDS,
	)
	col = bed_type_cvt[bed_type]

	if bed_type == :regular
		beds_data.NUM_NON_ICU_BEDS = beds_data.NUM_STAFFED_BEDS - beds_data.NUM_ICU_BEDS
	end

	filter!(row -> !ismissing(row[col]), beds_data)
	filter!(row -> row[col] > 0, beds_data)
	beds_data.selected_beds = beds_data[:,col]
	return beds_data
end

function definitivehc_aggregate(beds_data::DataFrame; locations::Array=[])
	beds_data_agg = combine(groupby(beds_data, :selected_location), selected_beds_agg = (:selected_beds => sum))
	if !isempty(locations)
		missing_locs = setdiff(locations, beds_data_agg.selected_location)
		for loc in missing_locs push!(beds_data_agg, Dict(:selected_location => loc, :selected_beds_agg => 0)) end
	end
	sort!(beds_data_agg, :selected_location)
    return Float32.(beds_data_agg.selected_beds_agg)
end

end;
