module NurseData

using CSV
using Dates
using DataFrames

export num_nurses, num_nurses_from_employment, num_nurses_from_beds

basepath = normpath(@__DIR__, "../../")

########################
####### Nurses #########
########################

let
	global nurse_data_employment
	nurse_data_employment = CSV.read(joinpath(basepath, "data/nurses/deaggregated_by_hospital_beds.csv"), copycols=true)
    nurse_data_employment = by(nurse_data_employment, :state, :weighted_emp_distribution => sum)
	sort!(nurse_data_employment, :state)
end

function num_nurses_from_employment(states::Array{String,1})
    @assert states == sort(states)
    @assert !("HI" in states)

    nurse_data_local = filter(row -> row.state in states, nurse_data_employment)
    return Float32.(nurse_data_local.weighted_emp_distribution_sum .* (1_713_120 / 2_982_280))
end
num_nurses = num_nurses_from_employment

let
	global nurse_data_beds

	nurse_data_beds = CSV.read(joinpath(basepath, "data/hospitals/Definitive_Healthcare__USA_Hospital_Beds.csv"), copycols=true)
	filter!(row -> !(row.HOSPITAL_TYPE in ["Psychiatric Hospital", "Rehabilitation Hospital"]), nurse_data_beds)
	filter!(row -> !(ismissing(row.NUM_STAFFED_BEDS) || ismissing(row.NUM_ICU_BEDS) || ismissing(row.HQ_STATE)), nurse_data_beds)
	filter!(row -> (row.NUM_STAFFED_BEDS > 0) && (row.NUM_ICU_BEDS > 0), nurse_data_beds)

	nurse_data_beds = by(nurse_data_beds, :HQ_STATE, [:NUM_STAFFED_BEDS => sum, :NUM_ICU_BEDS => sum])
	nurse_data_beds.non_icu_beds_sum = nurse_data_beds.NUM_STAFFED_BEDS_sum - nurse_data_beds.NUM_ICU_BEDS_sum
	sort!(nurse_data_beds, :HQ_STATE)
end

function num_nurses_from_beds(
		states::Array{String,1};
		nurse_hrs_per_week::Real=36,
		nurses_per_bed_regular::Real=0.2,
		nurses_per_bed_icu::Real=0.5
)
    @assert states == sort(states)
	nurse_data_local = filter(row -> row.HQ_STATE in states, nurse_data_beds)
	nurse_data_local.est_nurses =
		  (nurses_per_bed_icu * nurse_data_local.NUM_ICU_BEDS_sum)
		+ (nurses_per_bed_regular * nurse_data_local.non_icu_beds_sum)
    return Float32.(nurse_data_local.est_nurses * (24*7 / nurse_hrs_per_week))
end

end;
