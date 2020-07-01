module HospitalizationData

using CSV
using Dates
using DataFrames

basepath = normpath(@__DIR__, "../../")

export hospitalizations


function hospitalizations(locations::Array, start_date::Date, end_date::Date;
		level::Symbol=:state, source::Symbol=:cdc,
		forecast_type::Symbol=:active, patient_type::Symbol=:all, bound_type::Symbol=:mean,
		hospitalized_days::Int=-1, active_from_admitted::Bool=false,
)
	if source == :cdc || (source == :gt && forecast_type == :active)
		@assert level == :state
		@assert forecast_type == :active
		@assert patient_type  == :all
		gt = load_cdc_gt()
		gt = cdc_gt_filter!(gt, locations, start_date, end_date)
		patients = cdc_gt_history(gt, locations, start_date, end_date, bound_type=bound_type)
	elseif source == :covidtracking || (source == :gt && forecast_type == :admitted)
		@assert level == :state
		@assert patient_type  == :all
		@assert bound_type    == :mean
		forecast = load_covidtracking()
		forecast = covidtracking_filter!(forecast, locations, start_date, end_date)
		patients = covidtracking_history(forecast, forecast_type=forecast_type)
	end

	return patients
end

########################
#### COVIDTracking #####
########################

function load_covidtracking()
	gt = CSV.read(joinpath(basepath, "data/hospitalizations/covidtracking/2020-06-05/states_daily.csv"), copycols=true, silencewarnings=true)
	gt.date = map(d -> Date(string(d), "yyyymmdd"), gt.date)
	return gt
end

function covidtracking_filter!(gt::DataFrame, locations::Array{String,1}, start_date::Date, end_date::Date)
	@assert locations == sort(locations)
	@assert Date(2020, 1, 22) <= start_date <= end_date <= Date(2020, 6, 6)

	filter!(row -> start_date <= row.date <= end_date, gt)
	filter!(row -> row.state in locations, gt)
	sort!(gt, [:state, :date])
	return gt
end

function covidtracking_history(gt::DataFrame; forecast_type::Symbol=:active, keepmissings::Bool=false)
	groups = groupby(gt, :state, sort=true)
	if forecast_type == :active
		patients = vcat([g.hospitalizedCurrently' for g in groups]...)
	elseif forecast_type == :admitted
		patients = vcat([g.hospitalizedIncrease' for g in groups]...)
	else
		error("Invalid forecast type for covidtracking")
	end
	if !keepmissings
		patients[ismissing.(patients)] .= 0
	end
	return Float32.(patients)
end

########################
######## CDC GT ########
########################

function load_cdc_gt()
	goodcols = Dict(
		:state => :state,
		:collectionDate  => :date,
		:ICUBeds_Occ_AnyPat_Est  => :icu_active,
		:ICUBeds_Occ_AnyPat_LoCI => :icu_active_lb,
		:ICUBeds_Occ_AnyPat_UpCI => :icu_active_ub,
		:InpatBeds_Occ_AnyPat_Est  => :all_active,
		:InpatBeds_Occ_AnyPat_LoCI => :all_active_lb,
		:InpatBeds_Occ_AnyPat_UpCI => :all_active_ub,
		:InpatBeds_Occ_COVID_Est  => :all_covid_active,
		:InpatBeds_Occ_COVID_LoCI => :all_covid_active_lb,
		:InpatBeds_Occ_COVID_UpCI => :all_covid_active_ub,
	)
	goodcolnames = sort(collect(keys(goodcols)),rev=true)

	fn = joinpath(basepath, "data/hospitalizations/cdc/2020-06-16.csv")
	gt = DataFrame(CSV.File(fn, skipto=3, dateformat="dduuuyyyy", select=goodcolnames))
	rename!(gt, goodcols)
	return gt
end

function cdc_gt_filter!(gt::DataFrame, locations::Array{String,1}, start_date::Date, end_date::Date)
	@assert locations == sort(locations)
	@assert Date(2020, 4, 1) <= start_date <= end_date <= Date(2020, 6, 2)

	filter!(row -> row.state in locations, gt)
	filter!(row -> start_date <= row.date <= end_date, gt)
	sort!(gt, [:state, :date])
	return gt
end

function cdc_gt_history(
		gt::DataFrame,
		locations::Array{String,1}, start_date::Date, end_date::Date;
		patient_type::Symbol=:all, bound_type::Symbol=:mean,
)
	colselect = Dict(
		(:all, :mean) => :all_covid_active,
		(:all, :lb)   => :all_covid_active_lb,
		(:all, :ub)   => :all_covid_active_ub,
	)
	col = colselect[(patient_type, bound_type)]

	gt_by_state = groupby(gt, :state, sort=true)
	gt_by_state_date = Dict(s.state[1] => Dict(zip(s.date, s[!,col])) for s in gt_by_state)
	patients = [haskey(gt_by_state_date[s], d) ? gt_by_state_date[s][d] : missing for s in locations, d in start_date:Day(1):end_date]
	@assert !any(ismissing.(patients))

	return Float32.(patients)
end

end;
