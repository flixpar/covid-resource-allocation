module TexasData

using CSV, JSON
using Dates
using DataFrames

projectbasepath = "../../"
include(joinpath(projectbasepath, "src/processing/LocalDataCommon.jl"))
include(joinpath(projectbasepath, "src/processing/GeographicData.jl"))


function load_data_tx(
		date_range,
		los_dist, pct_beds_available,
		focus_locations, locations_limit;
		use_rounding=false,
	)
	@assert focus_locations isa Array || focus_locations isa Int

	rounddata = use_rounding ? x -> round.(Int, x) : identity

	rawdata = DataFrame(CSV.File(joinpath(projectbasepath, "data/local/texas/tx_tsa_hospitalizations.csv")))

	locations = sort(unique(rawdata.tsa_name))
	locations_limit = isnothing(locations_limit) ? length(locations) : locations_limit
	if focus_locations isa Int
		focus_locations = locations[1:focus_locations]
	end
	locations = vcat(focus_locations, setdiff(locations, focus_locations))
	locations = locations[1:min(length(locations), max(locations_limit, length(focus_locations)))]
	tsa_ids = [filter(row -> row.tsa_name == l, rawdata).tsa_id[1] for l in locations]

	data_common = (
		date_range = date_range,
		start_date = minimum(date_range),
		end_date = maximum(date_range),
		T = length(date_range),
		locations = locations,
		tsa_ids = tsa_ids,
		N = length(locations),
		focus_locations = focus_locations,
	)

	adj = compute_adj_tx(data_common)
	data_common = merge(data_common, (adj = adj,))

	data_dict = Dict((row.tsa_name, row.date) => (
		active_total          = row.patients_covid_total,
		active_icu            = row.patients_covid_icu,
		active_ward           = row.patients_covid_ward,
		capacity_allpat_total = row.beds_avail_total,
		capacity_allpat_icu   = row.beds_avail_icu,
		capacity_allpat_ward  = row.beds_avail_total - row.beds_avail_icu,
		allpat_total          = row.patients_all_total,
		allpat_icu            = row.patients_all_icu,
		allpat_ward           = row.patients_all_total - row.patients_all_icu,
		capacity_covid_total  = row.beds_avail_total - (row.patients_all_total - row.patients_covid_total),
		capacity_covid_icu    = row.beds_avail_icu - (row.patients_all_icu - row.patients_covid_icu),
		capacity_covid_ward   = (row.beds_avail_total - row.beds_avail_icu) - ((row.patients_all_total - row.patients_all_icu) - row.patients_covid_ward),
	) for row in eachrow(rawdata))
	start_date = minimum(date_range)

	data_allpat = merge(data_common, (
		active   = rounddata([data_dict[(c,d)].active_total for c in locations, d in date_range]),
		initial  = rounddata([data_dict[(c,start_date-Day(1))].active_total for c in locations]),
		capacity = rounddata([data_dict[(c,d)].capacity_covid_total for c in locations, d in date_range]),
		los_dist = los_dist.allpat,
		pct_beds_available = pct_beds_available.allpat,
	))
	adm_dis_all = LocalDataCommon.estimate_admitted_discharged(data_allpat, use_rounding=use_rounding)
	admitted_uncertainty_all = LocalDataCommon.estimate_admitted_uncertainty(data_allpat, adm_dis_all.admitted)
	beds_all = load_beds_tx(data_allpat, bedtype=:all, use_rounding=use_rounding)
	data_allpat = merge(data_allpat, adm_dis_all, (beds = beds_all, admitted_uncertainty = admitted_uncertainty_all))

	data_icu = merge(data_common, (
		active   = rounddata([data_dict[(c,d)].active_icu for c in locations, d in date_range]),
		initial  = rounddata([data_dict[(c,start_date-Day(1))].active_icu for c in locations]),
		capacity = rounddata([data_dict[(c,d)].capacity_covid_icu for c in locations, d in date_range]),
		los_dist = los_dist.icu,
		pct_beds_available = pct_beds_available.icu,
	))
	adm_dis_icu = LocalDataCommon.estimate_admitted_discharged(data_icu, use_rounding=use_rounding)
	admitted_uncertainty_icu = LocalDataCommon.estimate_admitted_uncertainty(data_icu, adm_dis_icu.admitted)
	beds_icu = load_beds_tx(data_icu, bedtype=:icu, use_rounding=use_rounding)
	data_icu = merge(data_icu, adm_dis_icu, (beds = beds_icu, admitted_uncertainty = admitted_uncertainty_icu))

	data_ward = merge(data_common, (
		active   = rounddata([data_dict[(c,d)].active_ward for c in locations, d in date_range]),
		initial  = rounddata([data_dict[(c,start_date-Day(1))].active_ward for c in locations]),
		capacity = rounddata([data_dict[(c,d)].capacity_covid_ward for c in locations, d in date_range]),
		los_dist = los_dist.ward,
		pct_beds_available = pct_beds_available.ward,
	))
	adm_dis_ward = LocalDataCommon.estimate_admitted_discharged(data_ward, use_rounding=use_rounding)
	admitted_uncertainty_ward = LocalDataCommon.estimate_admitted_uncertainty(data_ward, adm_dis_ward.admitted)
	beds_ward = load_beds_tx(data_ward, bedtype=:ward, use_rounding=use_rounding)
	data_ward = merge(data_ward, adm_dis_ward, (beds = beds_ward, admitted_uncertainty = admitted_uncertainty_ward))

	data = merge(data_common, (
		total = data_allpat,
		icu = data_icu,
		ward = data_ward,
	))

	data_carepathmodel = LocalDataCommon.build_carepaths_data(data)
	data = merge(data, (
		carepaths = data_carepathmodel,
	))

	return data
end;

function load_beds_tx(data; bedtype=:all, use_rounding=true)
	counties_list = JSON.parsefile(normpath(projectbasepath, "rawdata/texas/tsa_counties.json"));
	county_fips = [c["county_fips"] for c in counties_list]
	county_fips = parse.(Int, county_fips)

	beds_data = DataFrame(CSV.File(joinpath(projectbasepath, "data/hospitals/definitivehc.csv")))
	allowed_hospital_types = ["Critical Access Hospital", "Long Term Acute Care Hospital", "Short Term Acute Care Hospital"]
	filter!(row -> row.HQ_STATE == "TX", beds_data)
	filter!(row -> row.HOSPITAL_TYPE in allowed_hospital_types, beds_data)
	dropmissing!(beds_data, :FIPS)

	if bedtype == :all
		beds_col = :NUM_STAFFED_BEDS
	elseif bedtype == :icu
		beds_col = :NUM_ICU_BEDS
	elseif bedtype == :ward
		beds_col = :NUM_NON_ICU_BEDS
		beds_data.NUM_NON_ICU_BEDS = beds_data.NUM_STAFFED_BEDS - coalesce.(beds_data.NUM_ICU_BEDS, 0)
	end

	beds_data_agg = combine(groupby(beds_data, :FIPS), beds_col => (x -> sum(skipmissing(x))) => :beds)
	beds_data_dict = Dict(row.FIPS => row.beds for row in eachrow(beds_data_agg))
	beds_by_county = [(haskey(beds_data_dict, c) && !ismissing(beds_data_dict[c])) ? beds_data_dict[c] : 0 for c in county_fips]

	beds_by_tsa = Dict(zip(data.tsa_ids, zeros(Float64, data.N)))
	for (i, c) in enumerate(counties_list)
		tsa_id = c["tsa_id"]
		if haskey(beds_by_tsa, tsa_id)
			beds_by_tsa[tsa_id] += beds_by_county[i]
		end
	end

	beds = [haskey(beds_by_tsa, tsa_id) ? beds_by_tsa[tsa_id] : 0 for tsa_id in data.tsa_ids]
	beds = beds .* data.pct_beds_available

	if use_rounding
		beds = round.(Int, beds)
	end

	return beds
end;

function compute_adj_tx(data)
	# counties_list = JSON.parsefile(normpath(projectbasepath, "data/local/texas/tsa_counties.json"));
	# county_fips = [c["county_fips"] for c in counties_list]
	# county_fips = parse.(Int, county_fips)

	# adj = GeographicData.adjacencies(positions, level=:other, threshold=4.0)
	# return adj

	adj = GeographicData.adjacencies(data.locations, source=:fullyconnected, level=:other, threshold=4.0)
	return adj
end;

end;
