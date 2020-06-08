module NurseData

using CSV
using Dates
using DataFrames

export n_nurses

basepath = normpath(@__DIR__, "../../")

function n_nurses(states::Array{String,1}; source::Symbol=:ahrf)
	if source == :ahrf
		return num_nurses_from_ahrf(states)
	elseif source == :employment
		return num_nurses_from_employment(states)
	elseif source == :beds
		return num_nurses_from_beds(states)
	else
		error("Invalid nurses data source")
	end
end

########################
##### Employment #######
########################

function load_nurse_employment_data()
	nurse_data_employment = CSV.read(joinpath(basepath, "data/nurses/deaggregated_by_hospital_beds.csv"), copycols=true)
    nurse_data_employment = combine(groupby(nurse_data_employment, :state), :weighted_emp_distribution => sum)
	sort!(nurse_data_employment, :state)
	return nurse_data_employment
end

function num_nurses_from_employment(states::Array{String,1})
    @assert states == sort(states)
    @assert !("HI" in states)

	nurse_data_employment = load_nurse_employment_data()
    nurse_data_local = filter(row -> row.state in states, nurse_data_employment)
    return Float32.(nurse_data_local.weighted_emp_distribution_sum .* (1_713_120 / 2_982_280))
end

########################
######## Beds ##########
########################

function load_nurse_beds_data()
	nurse_data_beds = CSV.read(joinpath(basepath, "data/hospitals/definitivehc.csv"), copycols=true)
	filter!(row -> !(row.HOSPITAL_TYPE in ["Psychiatric Hospital", "Rehabilitation Hospital"]), nurse_data_beds)
	filter!(row -> !(ismissing(row.NUM_STAFFED_BEDS) || ismissing(row.NUM_ICU_BEDS) || ismissing(row.HQ_STATE)), nurse_data_beds)
	filter!(row -> (row.NUM_STAFFED_BEDS > 0) && (row.NUM_ICU_BEDS > 0), nurse_data_beds)

	nurse_data_beds = combine(groupby(nurse_data_beds, :HQ_STATE), [:NUM_STAFFED_BEDS => sum, :NUM_ICU_BEDS => sum])
	nurse_data_beds.non_icu_beds_sum = nurse_data_beds.NUM_STAFFED_BEDS_sum - nurse_data_beds.NUM_ICU_BEDS_sum
	sort!(nurse_data_beds, :HQ_STATE)

	return nurse_data_beds
end

function num_nurses_from_beds(
		states::Array{String,1};
		nurse_hrs_per_week::Real=36,
		nurses_per_bed_regular::Real=0.2,
		nurses_per_bed_icu::Real=0.5
)
    @assert states == sort(states)
	nurse_data_beds = load_nurse_beds_data()
	nurse_data_local = filter(row -> row.HQ_STATE in states, nurse_data_beds)
	nurse_data_local.est_nurses_per_shift =
		  (nurses_per_bed_icu * nurse_data_local.NUM_ICU_BEDS_sum)
		+ (nurses_per_bed_regular * nurse_data_local.non_icu_beds_sum)
	n_shifts = (24*7 / nurse_hrs_per_week)
	return Float32.(nurse_data_local.est_nurses_per_shift * n_shifts)
end

########################
######## AHRF ##########
########################

function load_nurse_ahrf_data()
	nurse_data = CSV.read(joinpath(basepath, "data/nurses/nurses_per_county.csv"), copycols=true)
	nurse_data_bystate = groupby(nurse_data, :state)
	nurse_data_ahrf = combine(nurse_data_bystate, [:registered_nurses => sum, :nurse_practitioners => sum, :tot_nurses => sum])
	sort!(nurse_data_ahrf, :state)
	return nurse_data_ahrf
end

function num_nurses_from_ahrf(states::Array{String,1})
	@assert states == sort(states)
	nurse_data_ahrf = load_nurse_ahrf_data()
	nurses_local = filter(row -> row.state in states, nurse_data_ahrf)
	nurses = nurses_local.registered_nurses_sum
	return Float32.(nurses)
end

end;
