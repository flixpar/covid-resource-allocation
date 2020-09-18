module Figures

using DataFrames
using CSV
using OrderedCollections: OrderedDict

using Distributions
using Dates

using Memoize
using ProgressMeter

using Gadfly
using Compose
import Cairo, Fontconfig

include("FiguresBase.jl")
using .FiguresBase


function compute_results(config, data, sent; use_rounding=false)
	N, T = data.N, data.T
	sent_null = zeros(Int, N, N, T)

	if use_rounding
		sent = round.(Int, sent)
	end

	model_title = titlecase(replace(config.experiment, "_" => " ")) * " Model"
	model_title = model_title == "Null Model" ? "No Transfers" : model_title

	active_patients = compute_active(sent, data.initial, data.discharged, data.admitted, data.los_dist, use_rounding=use_rounding)
	active_patients_nosent = compute_active(sent_null, data.initial, data.discharged, data.admitted, data.los_dist, use_rounding=use_rounding)

	overflow = compute_overflow(sent, data.initial, data.discharged, data.admitted, data.beds, data.los_dist, use_rounding=use_rounding)
	overflow_nosent = compute_overflow(sent_null, data.initial, data.discharged, data.admitted, data.beds, data.los_dist, use_rounding=use_rounding)

	load = compute_load(sent, data.initial, data.discharged, data.admitted, data.beds, data.los_dist, use_rounding=use_rounding)
	load_nosent = compute_load(sent_null, data.initial, data.discharged, data.admitted, data.beds, data.los_dist, use_rounding=use_rounding)

	results_table = DataFrame(
		date = permutedims(repeat(data.date_range, 1, N), (2,1))[:],
		location = repeat(data.locations, 1, T)[:],
		active_sent = active_patients[:],
		active_nosent = active_patients_nosent[:],
		overflow_sent = overflow[:],
		overflow_nosent = overflow_nosent[:],
		load_sent = load[:],
		load_nosent = load_nosent[:],
	)

	results = (
		sent = sent,
		active = active_patients,
		active_nosent = active_patients_nosent,
		overflow = overflow,
		overflow_nosent = overflow_nosent,
		load = load,
		load_nosent = load_nosent,
		table = results_table,
		model_title = model_title,
	)
	return results
end;

function compute_active(_sent, _initial, _discharged, _admitted, los_dist; use_rounding=false)
	N, _, T = size(_sent)
	_L = 1.0 .- cdf.(los_dist, 0:T)
	_active = [(
		_initial[i]
		- sum(_discharged[i,1:t])
		+ sum(_L[t-t₁+1] * (
			_admitted[i,t₁]
			- sum(_sent[i,:,t₁])
			+ sum(_sent[:,i,t₁])
		) for t₁ in 1:t)
		+ sum(_sent[i,:,t])
	) for i in 1:N, t in 1:T]
	if use_rounding
		_active = round.(Int, _active)
	end
	return _active
end;

function compute_overflow(_sent, _initial, _discharged, _admitted, _cap, los_dist; use_rounding=false)
	return max.(0, compute_active(_sent, _initial, _discharged, _admitted, los_dist, use_rounding=use_rounding) .- _cap)
end;

function compute_load(_sent, _initial, _discharged, _admitted, _cap, los_dist; use_rounding=false)
	return compute_active(_sent, _initial, _discharged, _admitted, los_dist, use_rounding=use_rounding) ./ _cap
end;

function compute_results_block(config, data, sent; use_rounding=false)
	G, N, T = data.G, data.N, data.T
	sent_null = zeros(Int, G, N, N, T)

	if use_rounding
		sent = round.(Int, sent)
	end

	model_title = titlecase(replace(config.experiment, "_" => " ")) * " Model"
	model_title = model_title == "Null Model" ? "No Transfers" : model_title

	active_patients = compute_active_block(sent, data.initial, data.discharged, data.admitted, data.los_bygroup, data.bedtype_bygroup, data.group_transfer_graph, use_rounding=use_rounding)
	active_patients_nosent = compute_active_block(sent_null, data.initial, data.discharged, data.admitted, data.los_bygroup, data.bedtype_bygroup, data.group_transfer_graph, use_rounding=use_rounding)

	overflow = compute_overflow_block(sent, data.initial, data.discharged, data.admitted, data.beds, data.los_bygroup, data.bedtype_bygroup, data.group_transfer_graph, use_rounding=use_rounding)
	overflow_nosent = compute_overflow_block(sent_null, data.initial, data.discharged, data.admitted, data.beds, data.los_bygroup, data.bedtype_bygroup, data.group_transfer_graph, use_rounding=use_rounding)

	load = compute_load_block(sent, data.initial, data.discharged, data.admitted, data.beds, data.los_bygroup, data.bedtype_bygroup, data.group_transfer_graph, use_rounding=use_rounding)
	load_nosent = compute_load_block(sent_null, data.initial, data.discharged, data.admitted, data.beds, data.los_bygroup, data.bedtype_bygroup, data.group_transfer_graph, use_rounding=use_rounding)

	results_table_icu = DataFrame(
		date = permutedims(repeat(data.date_range, 1, N), (2,1))[:],
		location = repeat(data.locations, 1, T)[:],
		active_sent = active_patients.icu[:],
		active_nosent = active_patients_nosent.icu[:],
		overflow_sent = overflow.icu[:],
		overflow_nosent = overflow_nosent.icu[:],
		load_sent = load.icu[:],
		load_nosent = load_nosent.icu[:],
	)
	results_table_ward = DataFrame(
		date = permutedims(repeat(data.date_range, 1, N), (2,1))[:],
		location = repeat(data.locations, 1, T)[:],
		active_sent = active_patients.ward[:],
		active_nosent = active_patients_nosent.ward[:],
		overflow_sent = overflow.ward[:],
		overflow_nosent = overflow_nosent.ward[:],
		load_sent = load.ward[:],
		load_nosent = load_nosent.ward[:],
	)

	results_icu = (
		sent = sent[2,:,:,:],
		active = active_patients.icu,
		active_nosent = active_patients_nosent.icu,
		overflow = overflow.icu,
		overflow_nosent = overflow_nosent.icu,
		load = load.icu,
		load_nosent = load_nosent.icu,
		table = results_table_icu,
		model_title = model_title,
	)
	results_ward = (
		sent = sent[4,:,:,:],
		active = active_patients.ward,
		active_nosent = active_patients_nosent.ward,
		overflow = overflow.ward,
		overflow_nosent = overflow_nosent.ward,
		load = load.ward,
		load_nosent = load_nosent.ward,
		table = results_table_ward,
		model_title = model_title,
	)

	results = (icu = results_icu, ward = results_ward)
	return results
end;

function compute_active_block(_sent, _initial, _discharged, _admitted, los, bed_types, transfer_graph; use_rounding=false)
	G, N, T = size(_admitted)
	B = length(unique(bed_types))

	F = [1.0 .- cdf(los[g], t) for g in 1:G, t in 0:T-1]
	f = [pdf(los[g], t) for g in 1:G, t in 0:T-1]
	f = f ./ sum(f, dims=2)

	groups_bybedtype = [sort(findall(x -> x == b, bed_types)) for b in 1:B]

	@memoize q(g,i,t) = (
		_admitted[g,i,t] + tr(g,i,t) + sum(_sent[g,j,i,t] - _sent[g,i,j,t] for j in 1:N)
	)
	@memoize l(g,i,t) = (
		_discharged[g,i,t] + sum(f[g,t-t₁+1] * q(g,i,t₁) for t₁ in 1:t)
	)
	@memoize tr(g,i,t) = (
		any(transfer_graph[:,g]) ?
			sum(l(g₁,i,t) for g₁ in findall(transfer_graph[:,g]))
			: 0
	)
	active_bygroup = [(
		_initial[g,i] + sum(_sent[g,i,:,t]) + sum((q(g,i,t₁) - l(g,i,t₁)) for t₁ in 1:t)
	) for g in 1:G, i in 1:N, t in 1:T]

	active_bybedtype = [sum(active_bygroup[g,i,t] for g in groups_bybedtype[b]) for b in 1:B, i in 1:N, t in 1:T]

	if use_rounding
		active_bybedtype = round.(Int, active_bybedtype)
	end

	_active = (icu = active_bybedtype[1,:,:], ward = active_bybedtype[2,:,:])
	return _active
end;

function compute_overflow_block(_sent, _initial, _discharged, _admitted, _cap, los, bed_types, transfer_graph; use_rounding=false)
	_active = compute_active_block(_sent, _initial, _discharged, _admitted, los, bed_types, transfer_graph, use_rounding=use_rounding)
	return (
		icu = max.(0, _active.icu .- _cap[1,:]),
		ward = max.(0, _active.ward .- _cap[2,:]),
	)
end;

function compute_load_block(_sent, _initial, _discharged, _admitted, _cap, los, bed_types, transfer_graph; use_rounding=false)
	_active = compute_active_block(_sent, _initial, _discharged, _admitted, los, bed_types, transfer_graph, use_rounding=use_rounding)
	return (
		icu = _active.icu ./ _cap[1,:],
		ward = _active.ward ./ _cap[2,:],
	)
end;

skipbad(x) = filter(y -> !(ismissing(y) || isnothing(y) || isnan(y) || isinf(y)), x);

function plot_metrics(config, data, results; bedtype=:all, display=true, save=true)
	N, T = data.N, data.T

	sent_out = sum(results.sent, dims=2)[:,1,:]
	sent_in = sum(results.sent, dims=1)[1,:,:]
	tfr = sent_out + sent_in

	o = sum(results.overflow[i,t] for i in 1:N, t in 1:T)
	o_null = sum(results.overflow_nosent[i,t] for i in 1:N, t in 1:T)
	o_red = (o_null - o) / o_null

	total_patients = sum(data.initial) + sum(data.admitted)
	total_patient_days = sum(results.active_nosent)

	nonzero(x) = filter(!=(0), x)
	printmetric(m, x; pct=false, int=false) = println("$(m): $(int ? x : round((pct ? x*100 : x),digits=2))" * (pct ? "%" : ""))

	if display
		println("Region: $(config.region)")
		println("Level: $(config.alloc_level)")
		println("Period: $(data.start_date) to $(data.end_date)")
		println("Bed type: $(bedtype)")
		println()
		printmetric("Number of days", T, int=true)
		printmetric("Number of hospitals", N, int=true)
		printmetric("Number of hospital-days", N*T, int=true)
		println()
		printmetric("Total patients", sum(data.admitted) + sum(data.initial))
		printmetric("Total patient-days", sum(results.active_nosent))
		printmetric("Peak active patients", maximum(sum(results.active_nosent, dims=1)))
		println()
		printmetric("Total sent", sum(results.sent))
		printmetric("Max sent by hospital-day", maximum(results.sent))
		printmetric("Max sent by day", maximum(sum(results.sent, dims=[1,2])))
		printmetric("Max sent by hospital", maximum(sum(results.sent, dims=[2,3])))
		println()
		printmetric("Number of transfers", sum(results.sent .> 0), int=true)
		printmetric("Percent of hospital-days with a transfer", mean(tfr .> 0), pct=true)
		printmetric("Percent of days with a transfer", mean(sum(tfr, dims=1) .> 0), pct=true)
		printmetric("Percent of hospitals with a transfer", mean(sum(tfr, dims=2) .> 0), pct=true)
		println()
		printmetric("Overflow (without transfers)", o_null)
		printmetric("Overflow (with transfers)", o)
		printmetric("Overflow reduction", o_red, pct=true)
		println()
		printmetric("Percent overflow (without transfers)", o_null / total_patient_days, pct=true)
		printmetric("Percent overflow (with transfers)", o / total_patient_days, pct=true)
		printmetric("Overflow reduction", o_red, pct=true)
		println()
		printmetric("Number of hospital-days with an overflow (without transfers)", sum(results.overflow_nosent .> 0))
		printmetric("Number of hospital-days with an overflow (with transfers)", sum(results.overflow .> 0))
		printmetric("Percent of hospital-days with an overflow (without transfers)", mean(results.overflow_nosent .> 0), pct=true)
		printmetric("Percent of hospital-days with an overflow (with transfers)", mean(results.overflow .> 0), pct=true)
		println()
		printmetric("Maximum overflow by hospital-day (without transfers)", maximum(results.overflow_nosent))
		printmetric("Maximum overflow by hospital-day (with transfers)", maximum(results.overflow))
		println()
		printmetric("Maximum load by hospital-day (without transfers)", maximum(skipbad(results.load_nosent)))
		printmetric("Maximum load by hospital-day (with transfers)", maximum(skipbad(results.load)))
	end

	metrics = OrderedDict(
		"experiment date" => config.rundate,
		"experiment" => config.experiment,
		"region" => config.region,
		"allocation level" => config.alloc_level,
		"bed type" => bedtype,
		"start date" => data.start_date,
		"end date" => data.end_date,
		"number of days" => T,
		"number of hospitals" => N,
		"number of hospital-days" => N*T,
		"total patients" => sum(data.admitted) + sum(data.initial),
		"total patient-days" => sum(results.active_nosent),
		"peak active patients" => maximum(sum(results.active_nosent, dims=1)),
		"total patients transferred" => sum(results.sent),
		"max sent by hospital-day" => maximum(results.sent),
		"max sent by day" => maximum(sum(results.sent, dims=[1,2])),
		"max sent by hospital" => maximum(sum(results.sent, dims=[2,3])),
		"number of transfers" => sum(results.sent .> 0),
		"proportion of hospital-days with a transfer" => mean(tfr .> 0),
		"proportion of days with a transfer" => mean(sum(tfr, dims=1) .> 0),
		"proportion of hospitals with a transfer" => mean(sum(tfr, dims=2) .> 0),
		"percent overflow (without transfers)" => (o_null / total_patient_days),
		"percent overflow (with transfers)" => (o / total_patient_days),
		"overflow (without transfers)" => o_null,
		"overflow (with transfers)" => o,
		"overflow reduction" => o_red,
		"number of hospital-days with an overflow (without transfers)" => sum(results.overflow_nosent .> 0),
		"number of hospital-days with an overflow (with transfers)" => sum(results.overflow .> 0),
		"proportion of hospital-days with an overflow (without transfers)" => mean(results.overflow_nosent .> 0),
		"proportion of hospital-days with an overflow (with transfers)" => mean(results.overflow .> 0),
		"maximum overflow by hospital-day (without transfers)" => maximum(results.overflow_nosent),
		"maximum overflow by hospital-day (with transfers)" => maximum(results.overflow),
		"maximum load by hospital-day (without transfers)" => maximum(skipbad(results.load_nosent)),
		"maximum load by hospital-day (with transfers)" => maximum(skipbad(results.load)),
		"median non-zero transfer" => iszero(results.sent) ? 0 : median(nonzero(results.sent[:])),
		"mean non-zero transfer" => iszero(results.sent) ? 0 : mean(nonzero(results.sent[:])),
		"max non-zero transfer" => iszero(results.sent) ? 0 : maximum(nonzero(results.sent[:])),
	)
	metrics_df = DataFrame(metric=collect(keys(metrics)), value=collect(values(metrics)))
	metrics_str = metrics_totextable(metrics_df)

	config_metrics = OrderedDict(
		"experiment date" => config.rundate,
		"experiment" => config.experiment,
		"region" => config.region,
		"allocation level" => config.alloc_level,
		"bed type" => bedtype,
		"start date" => data.start_date,
		"end date" => data.end_date,
		"number of days" => T,
		"number of hospitals" => N,
		"number of hospital-days" => N*T,
	)
	config_metrics_df = DataFrame(metric=collect(keys(config_metrics)), value=collect(values(config_metrics)))
	config_metrics_str = metrics_totextable(config_metrics_df)

	roundmetric(x) = round(x, digits=1)
	to_percent(x) = string(roundmetric(x*100))*"%"
	metrics_fortable = OrderedDict(
		"experiment" => config.experiment,

		# result metrics

		"overflow" => roundmetric(o),
		"overflow reduction" => to_percent(o_red),

		"median non-zero overflow" => iszero(results.overflow) ? 0 : roundmetric(median(nonzero(results.overflow[:]))),
		"mean non-zero overflow" => iszero(results.overflow) ? 0 : roundmetric(mean(nonzero(results.overflow[:]))),
		"max non-zero overflow" => iszero(results.overflow) ? 0 : roundmetric(maximum(nonzero(results.overflow[:]))),

		"median load" => to_percent(median(skipbad(results.load[:]))),
		"mean load" => to_percent(mean(skipbad(results.load[:]))),
		"max load" => to_percent(maximum(skipbad(results.load[:]))),

		"number of hospital-days with an overflow" => roundmetric(sum(results.overflow .> 0)),
		"percent of hospital-days with an overflow" => to_percent(mean(results.overflow .> 0)),

		# "maximum overflow by hospital-day" => roundmetric(maximum(results.overflow)),
		# "maximum load by hospital-day" => roundmetric(maximum(skipbad(results.load))),

		# transfer metrics

		"total patients transferred" => roundmetric(sum(results.sent)),
		"percent of patients transferred" => to_percent(sum(results.sent) / total_patients),

		"median non-zero transfer" => iszero(results.sent) ? 0 : roundmetric(median(nonzero(results.sent[:]))),
		"mean non-zero transfer" => iszero(results.sent) ? 0 : roundmetric(mean(nonzero(results.sent[:]))),
		"max non-zero transfer" => iszero(results.sent) ? 0 : roundmetric(maximum(nonzero(results.sent[:]))),

		# "max sent by day" => roundmetric(maximum(sum(results.sent, dims=[1,2]))),
		# "max sent by hospital" => roundmetric(maximum(sum(results.sent, dims=[2,3]))),

		"percent of hospital-days with a transfer" => to_percent(mean(tfr .> 0)),
		"percent of days with a transfer" => to_percent(mean(sum(tfr, dims=1) .> 0)),
		"percent of hospitals with a transfer" => to_percent(mean(sum(tfr, dims=2) .> 0)),
	)
	metrics_table_df = DataFrame(metric=collect(keys(metrics_fortable)), value=collect(values(metrics_fortable)))
	metrics_table_str = metrics_totextable(metrics_table_df)

	if save
		bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))

		metrics_dir = "figures/$(config.region_abbrev)/$(config.rundate)/$(config.experiment)/metrics/"
		if !isdir(metrics_dir) mkpath(metrics_dir) end

		metrics_path_base = joinpath(metrics_dir, "metrics_$(config.experiment)$(bedtype_ext)")
		metrics_df |> CSV.write(metrics_path_base * ".csv")
		write(metrics_path_base * ".tex", metrics_str)

		metrics_table_path_base = joinpath(metrics_dir, "metrics_table_$(config.experiment)$(bedtype_ext)")
		metrics_table_df |> CSV.write(metrics_table_path_base * ".csv")
		write(metrics_table_path_base * ".tex", metrics_table_str)

		config_metrics_path_base = joinpath(metrics_dir, "config_$(config.experiment)$(bedtype_ext)")
		config_metrics_df |> CSV.write(config_metrics_path_base * ".csv")
		write(config_metrics_path_base * ".tex", config_metrics_str)
	end

	return
end;

function plot_estimates_total(config, data; bedtype=:all, display=true, save=true)
	active_computed = compute_active(
		zeros(Int, data.N, data.N, data.T),
		data.initial,
		data.discharged,
		data.admitted,
		data.los_dist,
	)
	colors = Scale.default_discrete_colors(3)[[1,3]]
	plt_active_computed = plot(
		layer(x=data.date_range, y=sum(data.active, dims=1)[:], Geom.point, Geom.line, style(default_color=colors[1])),
		layer(x=data.date_range, y=sum(active_computed, dims=1)[:], Geom.point, Geom.line, style(default_color=colors[2])),
		Scale.y_continuous(format=:plain),
		Guide.xlabel("Date"), Guide.ylabel("Active COVID Patients"), Guide.title("Total Active COVID Patients - Raw vs Computed"),
		Guide.manual_color_key("", ["Data", "Computed"], colors[[1,2]], shape=[Shape.square], size=[1.6mm]),
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		Scale.y_continuous(format=:plain),
		style(
			key_position=:none,
			minor_label_font_size=20px,
			major_label_font_size=25px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			key_label_font_size=16px,
			key_title_font_size=18px,
			key_label_font="CMU Serif",
			key_title_font="CMU Serif",
			background_color=colorant"white",
		),
	)
	if display
		plt_active_computed |> SVG(20cm, 10cm)
	end

	plt_admitted_computed = plot(
		layer(x=data.date_range, y=sum(data.admitted, dims=1)[:], Geom.point, Geom.line, style(default_color=colors[1])),
		Guide.xlabel("Date"), Guide.ylabel("Admitted Patients"),
		Guide.title("Total Admitted Patients - Estimated"),
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		Scale.y_continuous(format=:plain),
		style(
			key_position=:none,
			minor_label_font_size=20px,
			major_label_font_size=25px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			background_color=colorant"white",
		),
	)
	plt_discharged_computed = plot(
		layer(x=data.date_range, y=sum(data.discharged, dims=1)[:], Geom.point, Geom.line, style(default_color=colors[2])),
		Guide.xlabel("Date"), Guide.ylabel("Discharged Patients"),
		Guide.title("Total Discharge of Initial Patients - Estimated"),
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		Scale.y_continuous(format=:plain),
		style(
			key_position=:none,
			minor_label_font_size=20px,
			major_label_font_size=25px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			background_color=colorant"white",
		),
	)
	plt_data_computed = hstack(plt_admitted_computed, plt_discharged_computed)
	if display
		plt_data_computed |> SVG(32cm, 10cm)
	end

	bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))

	if save
		save_all(plt_active_computed, 20cm, 10cm, "active_computed"*bedtype_ext, config)
		save_all(plt_data_computed, 32cm, 10cm, "adm_dis_computed"*bedtype_ext, config)
	end
end;

function plot_overflow_distribution(config, data, results; bedtype=:all, display=true, save=true)
	colors = Scale.default_discrete_colors(4)[[1,3]]
	binwidth = 5
	overflow_nonzero = filter(x -> x > 0, results.overflow)
	overflow_nosent_nonzero = filter(x -> x > 0, results.overflow_nosent)
	nbins_sent = !isempty(overflow_nonzero) ? ceil(maximum(overflow_nonzero) / binwidth) : 1
	nbins_nosent = !isempty(overflow_nosent_nonzero) ? ceil(maximum(overflow_nosent_nonzero) / binwidth) : 1
	overflow_dist_plot = plot(
		layer(
			x=overflow_nonzero,
			Geom.histogram(
				bincount=nbins_sent,
				limits=(min=0, max=nbins_sent*binwidth)
			),
			style(default_color=colors[1], alphas=[0.7])
		),
		layer(
			x=overflow_nosent_nonzero,
			Geom.histogram(
				bincount=nbins_nosent,
				limits=(min=0, max=nbins_nosent*binwidth)
			),
			style(default_color=colors[2], alphas=[0.7])
		),
		Guide.title("Distribution of (Non-Zero) Overflow by Hospital-Day"),
		Guide.xlabel("Overflow"),
		Guide.ylabel("Count"),
		Guide.manual_color_key("", ["With Transfers", "Without Transfers"], colors, shape=[Shape.square], size=[1.6mm]),
		style(
			minor_label_font_size=18px,
			major_label_font_size=22px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			key_label_font_size=16px,
			key_title_font_size=18px,
			key_label_font="CMU Serif",
			key_title_font="CMU Serif",
			background_color=colorant"white",
		),
	)

	bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))

	if display
		overflow_dist_plot |> SVG(23cm, 10cm)
	end
	if save
		save_all(overflow_dist_plot, 23cm, 10cm, "overflow_dist"*bedtype_ext, config)
	end
end;

function plot_load_distribution(config, data, results; bedtype=:all, display=true, save=true)
	colors = Scale.default_discrete_colors(4)[[1,3]]
	binwidth = 0.2
	load_sent = skipbad(results.load)
	load_nosent = skipbad(results.load_nosent)
	nbins_sent = ceil(maximum(load_sent) / binwidth)
	nbins_nosent = ceil(maximum(load_nosent) / binwidth)
	load_dist_plot = plot(
		layer(
			x=load_sent,
			Geom.histogram(
				bincount=nbins_sent,
				limits=(min=0, max=nbins_sent*binwidth)
			),
			style(default_color=colors[1], alphas=[0.7])
		),
		layer(
			x=load_nosent,
			Geom.histogram(
				bincount=nbins_nosent,
				limits=(min=0, max=nbins_nosent*binwidth)
			),
			style(default_color=colors[2], alphas=[0.7])
		),
		layer(xintercept=[1.0], Geom.vline(color="red", size=1.0mm), order=10),
		Guide.title("Distribution of Load by Hospital-Day"),
		Guide.xlabel("Load"),
		Guide.ylabel("Count"),
		Guide.manual_color_key("", ["With Transfers", "Without Transfers"], colors, shape=[Shape.square], size=[1.6mm]),
		Coord.Cartesian(xmax=5.0),
		style(
			minor_label_font_size=18px,
			major_label_font_size=22px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			key_label_font_size=16px,
			key_title_font_size=18px,
			key_label_font="CMU Serif",
			key_title_font="CMU Serif",
			background_color=colorant"white",
		),
	)

	bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))

	if display
		load_dist_plot |> SVG(23cm, 10cm)
	end
	if save
		save_all(load_dist_plot, 23cm, 10cm, "load_dist"*bedtype_ext, config);
	end
end;

function plot_maxload_distribution(config, data, results; bedtype=:all, display=true, save=true)
	colors = Scale.default_discrete_colors(4)[[1,3]]
	binwidth = 0.2
	maxload_sent = skipbad(maximum(results.load, dims=2))
	maxload_nosent = skipbad(maximum(results.load_nosent, dims=2))
	nbins_sent = ceil(maximum(maxload_sent) / binwidth)
	nbins_nosent = ceil(maximum(maxload_nosent) / binwidth)
	maxload_dist_plot = plot(
		layer(
			x=maxload_sent,
			Geom.histogram(
				bincount=nbins_sent,
				limits=(min=0, max=nbins_sent*binwidth)
			),
			style(default_color=colors[1], alphas=[0.7])
		),
		layer(
			x=maxload_nosent,
			Geom.histogram(
				bincount=nbins_nosent,
				limits=(min=0, max=nbins_nosent*binwidth)
			),
			style(default_color=colors[2], alphas=[0.7])
		),
		layer(xintercept=[1.0], Geom.vline(color="red", size=1.0mm), order=10),
		Guide.title("Distribution of Maximum Load by Hospital"),
		Guide.xlabel("Maximum Load"),
		Guide.ylabel("Count"),
		Guide.manual_color_key("", ["With Transfers", "Without Transfers"], colors, shape=[Shape.square], size=[1.6mm]),
		style(
			minor_label_font_size=18px,
			major_label_font_size=22px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			key_label_font_size=16px,
			key_title_font_size=18px,
			key_label_font="CMU Serif",
			key_title_font="CMU Serif",
			background_color=colorant"white",
		),
	)

	bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))

	if display
		maxload_dist_plot |> SVG(23cm, 10cm)
	end
	if save
		save_all(maxload_dist_plot, 23cm, 10cm, "maxload_dist"*bedtype_ext, config)
	end
end;

function plot_active_total(config, data, results; bedtype=:all, display=true, save=true)
	if bedtype == :icu
		plot_title = "Total Active COVID Patients in ICU Beds in $(config.region)"
		fn = "active_total_icu"
	elseif bedtype == :ward
		plot_title = "Total Active COVID Patients in Ward Beds in $(config.region)"
		fn = "active_total_ward"
	else
		plot_title = "Total Active COVID Patients in $(config.region)"
		fn = "active_total"
	end

	overall_plt = Gadfly.plot(
		layer(
			x = data.date_range,
			y = sum(results.active_nosent, dims=1)[:],
			Geom.point, Geom.line,
		),
		layer(
			yintercept = [sum(data.beds)],
			Geom.hline(color="red", size=0.7mm),
		),
		Guide.xlabel("Date"), Guide.ylabel("Active Patients"),
		Coord.Cartesian(ymin=0),
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		Scale.y_continuous(format=:plain),
		Guide.title(plot_title),
		style(
			key_position=:none,
			minor_label_font_size=20px,
			major_label_font_size=25px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			background_color=colorant"white",
		),
	)

	if display
		overall_plt |> SVG(20cm, 12cm)
	end
	if save
		save_all(overall_plt, 20cm, 12cm, fn, config)
	end
end;

function plot_active(config, data, results; add_title=true, columns=3, display=true, save=true, subset=nothing, bedtype=:all)
	if subset == nothing
		locs = enumerate(data.locations)
	else
		ind = [findfirst(==(h), data.locations) for h in subset]
		locs = zip(ind, subset)
	end

	active_plot_list = []
	colors = Scale.default_discrete_colors(5)[[1,4]]
	for (h_idx, h_name) in locs
		plot_title = length(h_name) > 43 ? strip(h_name[1:40])*"..." : h_name
		plt = plot(
			layer(x=data.date_range, y=results.active_nosent[h_idx,:], Geom.point, Geom.line, style(default_color=colors[1])),
			layer(x=data.date_range, y=results.active[h_idx,:], Geom.point, Geom.line, style(default_color=colors[2])),
			layer(yintercept=[data.beds[h_idx]], Geom.hline(color="red", size=0.7mm)),
			Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
			Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
			Coord.Cartesian(ymin=0),
			Guide.xlabel("Date"), Guide.ylabel("Active Patients"),
			Guide.title(plot_title),
			style(
				key_position=:none,
				minor_label_font_size=16px,
				major_label_font_size=18px,
				minor_label_font="CMU Serif",
				major_label_font="CMU Serif",
				background_color=colorant"white",
			),
		)
		push!(active_plot_list, plt)
	end

	@assert columns == 2 || columns == 3
	plot_width = Dict(2 => 16, 3 => 12)[columns]
	active_plots, active_plots_w, active_plots_h = plot_grid(active_plot_list, columns, plot_width=plot_width, return_plot=true, display=false)

	active_key_colors = vcat(colors, colorant"red")
	active_key = make_horizontal_key(["Without Transfers", "With Transfers", "Capacity"], active_key_colors, height=0.5cm)

	if bedtype == :all
		plot_title = "Active COVID Patients With and Without Transfers"
	elseif bedtype == :icu
		plot_title = "Active COVID Patients in ICU Beds With and Without Transfers"
	elseif bedtype == :ward
		plot_title = "Active COVID Patients in Ward Beds With and Without Transfers"
	end

	active_plots, dy1 = add_key_top(active_key, active_plots)
	active_plots, dy2 = add_title ? add_title_top(plot_title, active_plots) : (active_plots, 0cm)
	active_plots = fill_background(active_plots)

	active_plots_w = active_plots_w
	active_plots_h = active_plots_h + dy1 + dy2

	if display
		active_plots |> SVG(active_plots_w, active_plots_h)
	end

	if save
		out_name = isnothing(subset) ? "active" : "active_subset"
		out_name = (bedtype == :all) ? out_name : (bedtype == :icu ? out_name*"_icu" : (bedtype == :ward ? out_name*"_ward" : out_name))
		save_all(active_plots, active_plots_w, active_plots_h, out_name, config)
	end
end;

function plot_load(config, data, results; display=true, save=true, subset=nothing, bedtype=:all)
	if subset == nothing
		results_table = results.table
		locations = data.locations
		nlocations = data.N
	else
		results_table = filter(row -> row.location in subset, results.table)
		locations = subset
		nlocations = length(subset)
	end

	## load plots ##
	load_plot_max = 3.0
	plt_nosent = Gadfly.plot(
		results_table, x=:date, y=:load_nosent, color=:location,
		Geom.line,
		Guide.xlabel("Date"), Guide.ylabel("Patient Load"),
		Guide.title("COVID Patient Load in $(config.region) - Without Transfers"),
		layer(ymin=[0.0], ymax=[1.0],           Geom.hband, alpha=[0.1], Theme(default_color="green"), order=-1),
		layer(ymin=[1.0], ymax=[load_plot_max], Geom.hband, alpha=[0.1], Theme(default_color="red"),   order=-1),
		Coord.cartesian(xmin=data.start_date+Day(1), xmax=data.end_date-Day(1), ymin=0, ymax=load_plot_max),
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		style(
			key_position=:none,
			minor_label_font_size=20px,
			major_label_font_size=25px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
		),
	)
	plt_sent = Gadfly.plot(
		results_table, x=:date, y=:load_sent, color=:location,
		Geom.line,
		Guide.xlabel("Date"), Guide.ylabel("Patient Load"),
		Guide.title("COVID Patient Load in $(config.region) - With Transfers"),
		layer(ymin=[0.0], ymax=[1.0],           Geom.hband, alpha=[0.1], Theme(default_color="green"), order=-1),
		layer(ymin=[1.0], ymax=[load_plot_max], Geom.hband, alpha=[0.1], Theme(default_color="red"),   order=-1),
		Coord.cartesian(xmin=data.start_date+Day(1), xmax=data.end_date-Day(1), ymin=0, ymax=load_plot_max),
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		style(
			key_position=:none,
			minor_label_font_size=20px,
			major_label_font_size=25px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
		),
	)

	## load key ##
	load_key_labels = map(h_name -> length(h_name) > 53 ? strip(h_name[1:50])*"..." : h_name, locations)
	load_key, load_key_h = make_key(load_key_labels, Scale.default_discrete_colors(nlocations), line_height=0.5cm, ncolumns=3, shape=:circle)

	load_plts_withkey = compose(
		context(0,0,1,1),
		compose(
			context(0, 0, 1w, 1h - load_key_h),
			compose(context(0, 0, 0.5w, 1cy), render(plt_nosent)),
			compose(context(0.5w, 0, 0.5w, 1cy), render(plt_sent)),
		),
		compose(
			context(0, 1h - load_key_h, 1w, load_key_h),
			load_key,
		),
	)

	load_plts_withkey = fill_background(load_plts_withkey)

	if display
		load_plts_withkey |> SVG(40cm, 14cm + load_key_h)
	end

	if save
		out_name = isnothing(subset) ? "load_bylocation" : "load_bylocation_subset"
		out_name = (bedtype == :all) ? out_name : (bedtype == :icu ? out_name*"_icu" : (bedtype == :ward ? out_name*"_ward" : out_name))
		save_all(load_plts_withkey, 40cm, 14cm + load_key_h, out_name, config)
	end
end;

function plot_sent_total(config, data, results; bedtype=:all, display=true, save=true)
	sent_plot = plot(
		x=data.date_range,
		y=sum(results.sent, dims=[1,2])[:],
		Geom.point, Geom.line,
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		Coord.Cartesian(ymin=0),
		Guide.xlabel("Date"), Guide.ylabel("Patient Transfers"),
		Guide.title("Total Patient Transfers"),
		style(
			key_position=:none,
			minor_label_font_size=16px,
			major_label_font_size=18px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			background_color=colorant"white",
		),
	)

	bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))

	if display
		sent_plot |> SVG(20cm, 10cm)
	end
	if save
		save_all(sent_plot, 20cm, 10cm, "transfers_total"*bedtype_ext, config)
	end
end;

function plot_transfers(config, data, results; add_title=true, columns=3, display=true, save=true, subset=nothing, bedtype=:all)
	if subset == nothing
		locs = collect(enumerate(data.locations))
	else
		ind = [findfirst(==(h), data.locations) for h in subset]
		locs = zip(ind, subset)
	end

	transfer_plot_list = []
	colors = [colorant"seagreen4", colorant"firebrick1"]
	for (h_idx, h_name) in locs
		plot_title = length(h_name) > 43 ? strip(h_name[1:40])*"..." : h_name
		plt = plot(
			layer(x=data.date_range, y=sum(results.sent[:,h_idx,:], dims=1)[:], Geom.point, Geom.line, style(default_color=colors[1])),
			layer(x=data.date_range, y=sum(results.sent[h_idx,:,:], dims=1)[:], Geom.point, Geom.line, style(default_color=colors[2])),
			Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
			Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
			Coord.Cartesian(ymin=0),
			Guide.xlabel("Date"), Guide.ylabel("Patients"),
			Guide.title(plot_title),
			style(
				key_position=:none,
				minor_label_font_size=16px,
				major_label_font_size=18px,
				minor_label_font="CMU Serif",
				major_label_font="CMU Serif",
				background_color=colorant"white",
			),
		)
		push!(transfer_plot_list, plt)
	end

	@assert columns == 2 || columns == 3
	plot_width = Dict(2 => 16, 3 => 12)[columns]
	transfer_plot, plt_w, plt_h = plot_grid(transfer_plot_list, columns, plot_width=plot_width, return_plot=true, display=false)

	transfer_key = make_horizontal_key(["Received", "Sent"], colors, height=0.5cm)

	transfer_plot, dy1 = add_key_top(transfer_key, transfer_plot)
	transfer_plot, dy2 = add_title ? add_title_top("Patients Sent and Received by Hospital", transfer_plot) : (transfer_plot, 0cm)
	transfer_plot = fill_background(transfer_plot)

	transfer_plot_w = plt_w
	transfer_plot_h = plt_h + dy1 + dy2

	if display
		transfer_plot |> SVG(transfer_plot_w, transfer_plot_h)
	end

	if save
		out_name = isnothing(subset) ? "transfers" : "transfers_subset"
		out_name = (bedtype == :all) ? out_name : (bedtype == :icu ? out_name*"_icu" : (bedtype == :ward ? out_name*"_ward" : out_name))
		save_all(transfer_plot, transfer_plot_w, transfer_plot_h, out_name, config)
	end
end;

function sample_random_walk(n=30, maxchange=0.1)
	dist = Uniform(-maxchange,maxchange)
	walk = zeros(Float64, n)
	walk[1] = rand(Uniform(-1,1))
	for i in 2:n
		z = walk[i-1] + rand(dist)
		while z > 1 || z < -1
			z = walk[i-1] + rand(dist)
		end
		walk[i] = z
	end
	return walk
end;

function sample_forecast(data)
	forecast = zeros(Float64, data.N, data.T)
	for i in 1:data.N
		μ = data.admitted[i,:]
		d₁ = data.admitted_uncertainty[i,:]
		d₂ = data.admitted_uncertainty[i,:]

		walk = sample_random_walk(data.T, 0.5)
		patient_walk = μ .+ (max.(0, walk) .* d₁) .+ (min.(0, walk) .* d₂)

		forecast[i,:] = patient_walk
	end
	return forecast
end;

function plot_robust_overflow_distribution(config, data, results; display=true, save=true, bedtype=:all, debug=false)
	sent_null = zeros(Float64, data.N, data.N, data.T)
	_compute_active(_sent, _admitted) = compute_active(_sent, data.initial, data.discharged, _admitted, data.los_dist)
	_compute_total_overflow(_sent, _admitted) = sum(compute_overflow(_sent, data.initial, data.discharged, _admitted, data.beds, data.los_dist))

	n_samples = debug ? 20 : 300
	overflow_sampled_robust = Array{Float64, 1}(undef, n_samples)
	overflow_sampled_nonrobust = Array{Float64, 1}(undef, n_samples)
	overflow_sampled_null = Array{Float64, 1}(undef, n_samples)
	@showprogress for i in 1:n_samples
		_admitted = sample_forecast(data)
		overflow_sampled_robust[i] = _compute_total_overflow(results.sent_robust, _admitted)
		overflow_sampled_nonrobust[i] = _compute_total_overflow(results.sent_nonrobust, _admitted)
		overflow_sampled_null[i] = _compute_total_overflow(sent_null, _admitted)
	end

	binwidth = 100
	nbins_robust = ceil((maximum(overflow_sampled_robust) - minimum(overflow_sampled_robust)) / binwidth)
	nbins_nonrobust = ceil((maximum(overflow_sampled_robust) - minimum(overflow_sampled_robust)) / binwidth)
	nbins_null = ceil((maximum(overflow_sampled_robust) - minimum(overflow_sampled_robust)) / binwidth)

	colors = Scale.default_discrete_colors(4)[[4,1,3]]
	# colors = Scale.default_discrete_colors(4)[[4,2,3]]

	plt = plot(
		layer(
			x=overflow_sampled_robust,
			Geom.histogram(bincount=nbins_robust, density=true),
			style(
				default_color=colors[1],
				alphas=[0.6],
			),
		),
		layer(
			x=overflow_sampled_nonrobust,
			Geom.histogram(bincount=nbins_nonrobust, density=true),
			style(
				default_color=colors[2],
				alphas=[0.6],
			),
		),
		layer(
			x=overflow_sampled_null,
			Geom.histogram(bincount=nbins_null, density=true),
			style(
				default_color=colors[3],
				alphas=[0.6],
			),
		),
		Scale.x_continuous(format=:plain),
		Coord.Cartesian(xmin=0),
		Guide.manual_color_key(
			"Model", ["Robust", "Non-Robust", "No Transfers"], colors,
			shape=[Shape.square], size=[8px],
		),
		Guide.xlabel("Overflow (Patient-Days)"), Guide.ylabel("Frequency"),
		Guide.title("Overflow Distribution by Model"),
		style(
			key_position=:none,
			minor_label_font_size=16px,
			major_label_font_size=18px,
			key_title_font_size=16px,
			key_label_font_size=16px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			key_title_font="CMU Serif",
			key_label_font="CMU Serif",
			background_color=colorant"white",
		),
	)

	if display
		plt |> SVG(26cm, 10cm)
	end

	if save
		bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))
		save_all(plt, 26cm, 10cm, "overflow_distribution_robust"*bedtype_ext, config)
	end
end;

function plot_active_robust(config, data, results; display=true, save=true, columns=3, add_title=false, bedtype=:all, subset=nothing)
	if subset == nothing
		locs = collect(enumerate(data.locations))
	else
		ind = [findfirst(==(h), data.locations) for h in subset]
		locs = collect(zip(ind, subset))
	end

	sent_null = zeros(Float64, data.N, data.N, data.T)
	_compute_active(_sent, _admitted) = compute_active(_sent, data.initial, data.discharged, _admitted, data.los_dist)

	n_samples = 5
	sampled_admitted = [sample_forecast(data) for _ in 1:n_samples]
	active_sampled_robust = [_compute_active(results.sent_robust, sampled_admitted[i]) for i in 1:n_samples]
	active_sampled_nonrobust = [_compute_active(results.sent_nonrobust, sampled_admitted[i]) for i in 1:n_samples]
	active_sampled_null = [_compute_active(sent_null, sampled_admitted[i]) for i in 1:n_samples]

	colors = Scale.default_discrete_colors(4)[[4,1,3]]
	# colors = Scale.default_discrete_colors(4)[[4,2,3]]

	plots = []
	for (i, h_name) in locs
		sample_layers = []
		for j in 1:n_samples
			push!(sample_layers, layer(
				x = data.date_range,
				y = active_sampled_robust[j][i,:],
				Geom.line,
				style(default_color=colors[1]),
			))
			push!(sample_layers, layer(
				x = data.date_range,
				y = active_sampled_nonrobust[j][i,:],
				Geom.line,
				style(default_color=colors[2]),
			))
			push!(sample_layers, layer(
				x = data.date_range,
				y = active_sampled_null[j][i,:],
				Geom.line,
				style(default_color=colors[3]),
			))
		end

		plot_title = length(h_name) > 43 ? strip(h_name[1:40])*"..." : h_name
		plt = plot(
			sample_layers...,
			layer(yintercept=[data.beds[i]], Geom.hline(color="red", size=0.7mm)),
			Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
			Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
			Coord.Cartesian(ymin=0),
			Guide.xlabel("Date"), Guide.ylabel("Active Patients"),
			Guide.title(plot_title),
			style(
				key_position=:none,
				minor_label_font_size=16px,
				major_label_font_size=18px,
				minor_label_font="CMU Serif",
				major_label_font="CMU Serif",
				background_color=colorant"white",
			),
		)
		push!(plots, plt)
	end

	@assert columns == 2 || columns == 3
	plot_width = Dict(2 => 16, 3 => 12)[columns]
	active_plots, active_plots_w, active_plots_h = plot_grid(plots, columns, plot_width=plot_width, return_plot=true, display=false)

	active_key_colors = vcat(colors, colorant"red")
	active_key = make_horizontal_key(["Robust", "Non-Robust", "No Transfers", "Capacity"], active_key_colors, height=0.5cm)

	if bedtype == :all
		plot_title = "Active COVID Patients By Model"
	elseif bedtype == :icu
		plot_title = "Active COVID Patients in ICU Beds By Model"
	elseif bedtype == :ward
		plot_title = "Active COVID Patients in Ward Beds By Model"
	end

	active_plots, dy1 = add_key_top(active_key, active_plots)
	active_plots, dy2 = add_title ? add_title_top(plot_title, active_plots) : (active_plots, 0cm)
	active_plots = fill_background(active_plots)

	active_plots_w = active_plots_w
	active_plots_h = active_plots_h + dy1 + dy2

	if display
		active_plots |> SVG(active_plots_w, active_plots_h)
	end

	if save
		out_name = isnothing(subset) ? "active_robust" : "active_robust_subset"
		out_name = (bedtype == :all) ? out_name : (bedtype == :icu ? out_name*"_icu" : (bedtype == :ward ? out_name*"_ward" : out_name))
		save_all(active_plots, active_plots_w, active_plots_h, out_name, config)
	end
end;

function plot_active_samples_notransfers(config, data, results; display=true, save=true, columns=3, add_title=true, bedtype=:all, subset=nothing)
	if subset == nothing
		locs = collect(enumerate(data.locations))
	else
		ind = [findfirst(==(h), data.locations) for h in subset]
		locs = collect(zip(ind, subset))
	end

	sent_null = zeros(Float64, data.N, data.N, data.T)
	_compute_active(_admitted) = compute_active(sent_null, data.initial, data.discharged, _admitted, data.los_dist)

	active_null_lb = _compute_active(data.admitted - data.admitted_uncertainty)
	active_null_ub = _compute_active(data.admitted + data.admitted_uncertainty)

	n_samples = 5
	sampled_admitted = [sample_forecast(data) for _ in 1:n_samples-1]
	insert!(sampled_admitted, 1, data.admitted)
	sampled_active = [_compute_active(sampled_admitted[i]) for i in 1:n_samples]

	colors = Scale.default_discrete_colors(n_samples)

	plots = []
	for (i, h_name) in locs
		sample_layers = []
		for j in 1:n_samples
			push!(sample_layers, layer(
				x = data.date_range,
				y = sampled_active[j][i,:],
				Geom.line,
				style(default_color=colors[j]),
				order=10,
			))
		end

		plot_title = length(h_name) > 43 ? strip(h_name[1:40])*"..." : h_name
		plt = plot(
			layer(
				x = data.date_range,
				ymin = active_null_lb[i,:],
				ymax = active_null_ub[i,:],
				Geom.ribbon,
				style(alphas=[0.5]),
			),

			sample_layers...,
			layer(yintercept=[data.beds[i]], Geom.hline(color="red", size=0.7mm)),

			Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
			Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
			Coord.Cartesian(ymin=0),
			Guide.xlabel("Date"), Guide.ylabel("Active Patients"),
			Guide.title(plot_title),
			style(
				key_position=:none,
				minor_label_font_size=16px,
				major_label_font_size=18px,
				minor_label_font="CMU Serif",
				major_label_font="CMU Serif",
				background_color=colorant"white",
			),
		)
		push!(plots, plt)
	end

	@assert columns == 2 || columns == 3
	plot_width = Dict(2 => 16, 3 => 12)[columns]
	active_plots, active_plots_w, active_plots_h = plot_grid(plots, columns, plot_width=plot_width, return_plot=true, display=false)

	if bedtype == :all
		plot_title = "Active COVID Patients Sampled from the Uncertainty Set"
	elseif bedtype == :icu
		plot_title = "Active COVID Patients in ICU Beds Sampled from the Uncertainty Set"
	elseif bedtype == :ward
		plot_title = "Active COVID Patients in Ward Beds Sampled from the Uncertainty Set"
	end

	active_plots, dy1 = add_title ? add_title_top(plot_title, active_plots) : (active_plots, 0cm)
	active_plots = fill_background(active_plots)

	active_plots_w = active_plots_w
	active_plots_h = active_plots_h + dy1

	if display
		active_plots |> SVG(active_plots_w, active_plots_h)
	end

	if save
		out_name = isnothing(subset) ? "active_samples" : "active_samples_subset"
		out_name = (bedtype == :all) ? out_name : (bedtype == :icu ? out_name*"_icu" : (bedtype == :ward ? out_name*"_ward" : out_name))
		save_all(active_plots, active_plots_w, active_plots_h, out_name, config)
	end
end;

end;