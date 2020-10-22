module PatientAllocationFigures

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

	capacity_byday = permutedims(repeat(data.capacity, 1, 1, data.T), (1,3,2))
	overflow_bycapacity = compute_overflow(sent, data.initial, data.discharged, data.admitted, capacity_byday, data.los_dist, use_rounding=use_rounding)
	overflow_bycapacity_nosent = compute_overflow(sent_null, data.initial, data.discharged, data.admitted, capacity_byday, data.los_dist, use_rounding=use_rounding)
	load_bycapacity = compute_load(sent, data.initial, data.discharged, data.admitted, capacity_byday, data.los_dist, use_rounding=use_rounding)
	load_bycapacity_nosent = compute_load(sent_null, data.initial, data.discharged, data.admitted, capacity_byday, data.los_dist, use_rounding=use_rounding)

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
		overflow_bycapacity = overflow_bycapacity,
		overflow_bycapacity_nosent = overflow_bycapacity_nosent,
		load_bycapacity = load_bycapacity,
		load_bycapacity_nosent = load_bycapacity_nosent,
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

function compute_metrics(config, data, results, bedtype=:all)
	N, T = data.N, data.T

	sent_out = sum(results.sent, dims=2)[:,1,:]
	sent_in = sum(results.sent, dims=1)[1,:,:]
	tfr = sent_out + sent_in

	o = sum(results.overflow[i,t] for i in 1:N, t in 1:T)
	o_null = sum(results.overflow_nosent[i,t] for i in 1:N, t in 1:T)
	o_red = (o_null - o) / o_null

	total_patients = sum(data.initial) + sum(data.admitted)

	nonzero(x) = filter(!=(0), x)

	metrics = [
		("experiment",                                     :config,    :str,  true,   config.experiment),
		("experiment date",                                :config,    :str,  false,  config.rundate),
		("region",                                         :config,    :str,  false,  config.region),
		("allocation level",                               :config,    :str,  false,  config.alloc_level),
		("bed type",                                       :config,    :str,  false,  bedtype),
		("start date",                                     :config,    :dt,   false,  data.start_date),
		("end date",                                       :config,    :dt,   false,  data.end_date),

		("number of days",                                 :data,      :int,  false,  T),
		("number of hospitals",                            :data,      :int,  false,  N),
		("number of hospital-days",                        :data,      :int,  false,  N*T),

		("total patients",                                 :data,      :int,  false,  sum(data.admitted) + sum(data.initial)),
		("total patient-days",                             :data,      :int,  false,  sum(results.active_nosent)),
		("peak active patients",                           :data,      :int,  false,  maximum(sum(results.active_nosent, dims=1))),

		("overflow",                                       :results,   :int,  true,   o),
		("overflow reduction",                             :results,   :pct,  true,   o_red),

		("median non-zero overflow",                       :results,   :int,  true,   iszero(results.overflow) ? 0 : median(nonzero(results.overflow[:]))),
		("mean non-zero overflow",                         :results,   :flt,  true,   iszero(results.overflow) ? 0 : mean(nonzero(results.overflow[:]))),
		("maximum overflow",                               :results,   :int,  true,   maximum(results.overflow)),

		("median load",                                    :results,   :flt,  true,   median(skipbad(results.load[:]))),
		("mean load",                                      :results,   :flt,  true,   mean(skipbad(results.load[:]))),
		("max load",                                       :results,   :flt,  true,   maximum(skipbad(results.load[:]))),

		("number of hospital-days with an overflow",       :results,   :int,  true,   sum(results.overflow .> 0)),
		("percent of hospital-days with an overflow",      :results,   :pct,  true,   mean(results.overflow .> 0)),
		("proportion of hospital-days with an overflow",   :results,   :prop, false,  mean(results.overflow .> 0)),

		("total patients transferred",                     :transfer,  :int,  true,   sum(results.sent)),
		("percent of patients transferred",                :transfer,  :pct,  true,   sum(results.sent) / total_patients),

		("percent of hospital-days with a transfer",       :transfer,  :pct,  true,   mean(tfr .> 0)),
		("percent of days with a transfer",                :transfer,  :pct,  true,   mean(sum(tfr, dims=1) .> 0)),
		("percent of hospitals with a transfer",           :transfer,  :pct,  true,   mean(sum(tfr, dims=2) .> 0)),

		("median non-zero transfer",                       :transfer,  :int,  true,   iszero(results.sent) ? 0 : median(nonzero(results.sent[:]))),
		("mean non-zero transfer",                         :transfer,  :flt,  true,   iszero(results.sent) ? 0 : mean(nonzero(results.sent[:]))),
		("max non-zero transfer",                          :transfer,  :int,  true,   iszero(results.sent) ? 0 : maximum(nonzero(results.sent[:]))),

		("number of transfers",                            :transfer,  :int,  false,  sum(results.sent .> 0)),
		("proportion of patients transferred",             :transfer,  :prop, false,  sum(results.sent) / total_patients),

		("max sent by hospital-day",                       :transfer,  :int,  false,  maximum(results.sent)),
		("max sent by day",                                :transfer,  :int,  false,  maximum(sum(results.sent, dims=[1,2]))),
		("max sent by hospital",                           :transfer,  :int,  false,  maximum(sum(results.sent, dims=[2,3]))),

		("proportion of hospital-days with a transfer",    :transfer,  :prop, false,  mean(tfr .> 0)),
		("proportion of days with a transfer",             :transfer,  :prop, false,  mean(sum(tfr, dims=1) .> 0)),
		("proportion of hospitals with a transfer",        :transfer,  :prop, false,  mean(sum(tfr, dims=2) .> 0)),
	]

	return metrics
end;

function plot_metrics(config, data, results; bedtype=:all, display=true, save=true)
	metrics = compute_metrics(config, data, results, bedtype)

	roundmetric(x) = round(x, digits=1)
	to_percent(x) = string(roundmetric(x*100))*"%"
	format_value(x, m_type) = begin
		if m_type == :pct
			return to_percent(x)
		elseif m_type == :flt
			return string(roundmetric(x))
		elseif m_type == :prop
			return string(round(x, digits=3))
		elseif m_type == :int && (x isa Float64 || x isa Float32)
			return round(Int, x)
		else
			return string(x)
		end
	end
	printmetric(m, x; m_type=:str, m_width=0) = begin
		x = format_value(x, m_type)
		m = m * ":"
		m = (length(m) < m_width+1) ? rpad(m, m_width+1) : m
		println("$(m) $(x)")
	end

	if display
		m_cat_prev = nothing
		m_width = maximum([length(m[1]) for m in metrics]) + 2
		for (m_name, m_cat, m_type, _, m_val) in metrics
			if (m_cat_prev != m_cat) println() end
			m_cat_prev = m_cat
			printmetric(m_name, m_val, m_type=m_type, m_width=m_width)
		end
	end

	all_metrics = DataFrame(
		metric=[m[1] for m in metrics],
		value=[format_value(m[5], m[3]) for m in metrics],
	)
	config_metrics = DataFrame(
		metric=[m[1] for m in metrics if m[2] == :config],
		value=[format_value(m[5], m[3]) for m in metrics if m[2] == :config],
	)
	compare_metrics = DataFrame(
		metric=[m[1] for m in metrics if m[4]],
		value=[format_value(m[5], m[3]) for m in metrics if m[4]],
	)

	if save
		bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))
		ext = "$(config.experiment)$(bedtype_ext)"

		metrics_dir = "$(config.results_basepath)/$(config.region_abbrev)/$(config.rundate)/$(config.experiment)/metrics/"
		if !isdir(metrics_dir) mkpath(metrics_dir) end

		figures_dir = "$(config.paperfigures_basepath)/$(config.region_abbrev)/$(config.experiment)/"
		if !isdir(figures_dir) mkpath(figures_dir) end

		all_metrics |> CSV.write(joinpath(metrics_dir, "metrics_all_$(ext).csv"))
		all_metrics_tex = metrics_totextable(all_metrics)
		write(joinpath(metrics_dir, "metrics_all_$(ext).tex"), all_metrics_tex)
		write(joinpath(figures_dir, "metrics_all$(bedtype_ext).tex"), all_metrics_tex)

		compare_metrics |> CSV.write(joinpath(metrics_dir, "metrics_$(ext).csv"))
		compare_metrics_tex = metrics_totextable(compare_metrics, header=true)
		write(joinpath(metrics_dir, "metrics_$(ext).tex"), compare_metrics_tex)
		write(joinpath(figures_dir, "metrics$(bedtype_ext).tex"), compare_metrics_tex)

		config_metrics |> CSV.write(joinpath(metrics_dir, "config_$(ext).csv"))
		config_metrics_tex = metrics_totextable(config_metrics)
		write(joinpath(metrics_dir, "config_$(ext).tex"), config_metrics_tex)
		write(joinpath(figures_dir, "config$(bedtype_ext).tex"), config_metrics_tex)
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
		Guide.xlabel("Date"), Guide.ylabel("Active COVID Patients"), Guide.title("Total Active COVID Patients - Raw vs Computed"),
		Guide.manual_color_key("", ["Data", "Computed"], colors[[1,2]], shape=[Shape.square], size=[1.6mm]),
		Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
		Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
		Scale.y_continuous(format=:plain),
		Coord.cartesian(ymin=0),
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

function plot_load(config, data, results; key_columns=3, display=true, save=true, subset=nothing, bedtype=:all, multiple_capacity=false)
	if !multiple_capacity
		return _plot_load(config, data, results; key_columns=key_columns, display=display, save=save, subset=subset, bedtype=bedtype, capacity_name=nothing)
	end

	C = size(data.capacity, 2)
	_results_table = deepcopy(results.table)
	for c in 1:C
		cap_abbrev = data.capacity_names_abbrev[c]
		cap_name = data.capacity_names_full[c]

		_results_table[!,:load_sent] = results.load_bycapacity[:,:,c][:]
		_results_table[!,:load_nosent] = results.load_bycapacity_nosent[:,:,c][:]
		_results = merge(results, (table = _results_table,))

		_plot_load(config, data, _results; key_columns=key_columns, display=display, save=save, subset=subset, bedtype=bedtype, capacity_name=cap_abbrev)
	end
end

function _plot_load(config, data, results; key_columns=3, display=true, save=true, subset=nothing, bedtype=:all, capacity_name=nothing)
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
		layer(
			results_table, x=:date, y=:load_nosent, color=:location,
			Geom.line,
		),
		Guide.xlabel("Date"), Guide.ylabel("Patient Load"),
		Guide.title("COVID Patient Load in $(config.region) - Without Transfers"),
		layer(ymin=[0.0], ymax=[1.0],           Geom.hband, alpha=[0.1], style(default_color=colorant"green"), order=-1),
		layer(ymin=[1.0], ymax=[load_plot_max], Geom.hband, alpha=[0.1], style(default_color=colorant"red"),   order=-1),
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
		layer(
			results_table, x=:date, y=:load_sent, color=:location,
			Geom.line,
		),
		Guide.xlabel("Date"), Guide.ylabel("Patient Load"),
		Guide.title("COVID Patient Load in $(config.region) - With Transfers"),
		layer(ymin=[0.0], ymax=[1.0],           Geom.hband, alpha=[0.1], style(default_color=colorant"green"), order=-1),
		layer(ymin=[1.0], ymax=[load_plot_max], Geom.hband, alpha=[0.1], style(default_color=colorant"red"),   order=-1),
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
	load_key, load_key_h = make_key(load_key_labels, Scale.default_discrete_colors(nlocations), line_height=0.6cm, ncolumns=key_columns, shape=:circle)

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
		out_name = isnothing(capacity_name) ? out_name : (out_name * "_" * capacity_name)
		save_all(load_plts_withkey, 40cm, 14cm + load_key_h, out_name, config)
	end

	return
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

function plot_metrics_compare(shared_config, data, results_list; bedtype=:all, display_table=true, save=true)

	roundmetric(x) = round(x, digits=1)
	to_percent(x) = string(roundmetric(x*100))*"%"
	format_value(x, m_type) = begin
		if m_type == :pct
			return to_percent(x)
		elseif m_type == :flt
			return string(roundmetric(x))
		elseif m_type == :prop
			return string(round(x, digits=3))
		elseif m_type == :int && (x isa Float64 || x isa Float32)
			return round(Int, x)
		else
			return string(x)
		end
	end

	metrics_table = DataFrame(metric=[])
	for (exp_name, _results) in results_list
		_config = merge(shared_config, (experiment = exp_name,))
		_metrics = compute_metrics(_config, data, _results)
		_compare_metrics = DataFrame(
			metric=[m[1] for m in _metrics if m[4]],
			value=[format_value(m[5], m[3]) for m in _metrics if m[4]],
		)
		rename!(_compare_metrics, "value" => exp_name)
		metrics_table = outerjoin(metrics_table, _compare_metrics, on=:metric)
	end

	if display_table
		display(metrics_table)
	end

	if save
		bedtype_ext = (bedtype == :all) ? "" : (bedtype == :icu ? "_icu" : (bedtype == :ward ? "_ward" : ""))

		metrics_dir = "$(shared_config.results_basepath)/$(shared_config.region_abbrev)/$(shared_config.rundate)/compare/metrics/"
		if !isdir(metrics_dir) mkpath(metrics_dir) end

		out_path_base = joinpath(metrics_dir, "metrics_compare$(bedtype_ext)")
		metrics_table |> CSV.write(out_path_base * ".csv")
		write(out_path_base * ".tex", metrics_totextable(metrics_table, header=true))
	end

	return
end;

function plot_figures_list(config)
end;

end;
