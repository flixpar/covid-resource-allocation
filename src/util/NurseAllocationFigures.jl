module NurseAllocationFigures

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


function compute_results_nurses(config, data, sent; use_rounding=false)
	N, T = data.N, data.T
	sent_null = zeros(Int, N, N, T)

	if use_rounding
		sent = round.(Int, sent)
	end

	model_title = titlecase(replace(config.experiment, "_" => " ")) * " Model"
	model_title = model_title == "Null Model" ? "No Transfers" : model_title

	active_nurses = compute_active_nurses(sent, data.nurses, use_rounding=use_rounding)
	active_nurses_nosent = compute_active_nurses(sent_null, data.nurses, use_rounding=use_rounding)

	shortage = compute_nurse_shortage(sent, data.nurses, data.demand, use_rounding=use_rounding)
	shortage_nosent = compute_nurse_shortage(sent_null, data.nurses, data.demand, use_rounding=use_rounding)

	load = compute_nurse_load(sent, data.nurses, data.demand, use_rounding=use_rounding)
	load_nosent = compute_nurse_load(sent_null, data.nurses, data.demand, use_rounding=use_rounding)

	results_table = DataFrame(
		date = permutedims(repeat(data.date_range, 1, N), (2,1))[:],
		location = repeat(data.locations, 1, T)[:],
		active_sent = active_nurses[:],
		active_nosent = active_nurses_nosent[:],
		shortage_sent = shortage[:],
		shortage_nosent = shortage_nosent[:],
		load_sent = load[:],
		load_nosent = load_nosent[:],
	)

	results = (
		sent = sent,
		active = active_nurses,
		active_nosent = active_nurses_nosent,
		shortage = shortage,
		shortage_nosent = shortage_nosent,
		load = load,
		load_nosent = load_nosent,
		table = results_table,
		model_title = model_title,
	)
	return results
end;

function compute_active_nurses(_sent, _nurses; use_rounding=false)
	N, _, T = size(_sent)
	_active = [(
		_nurses[i]
		- sum(_sent[i,:,1:t])
		+ sum(_sent[:,i,1:t])
	) for i in 1:N, t in 1:T]
	if use_rounding
		_active = round.(Int, _active)
	end
	return _active
end;

function compute_nurse_shortage(_sent, _nurses, _demand; use_rounding=false)
	return max.(0, compute_active_nurses(_sent, _nurses, use_rounding=use_rounding) .- _demand)
end;

function compute_nurse_load(_sent, _nurses, _demand; use_rounding=false)
	return compute_active_nurses(_sent, _nurses, use_rounding=use_rounding) ./ _demand
end;

function plot_metrics(config, data, results; bedtype=:all, display=true, save=true)
	N, T = data.N, data.T

	sent_out = sum(results.sent, dims=2)[:,1,:]
	sent_in = sum(results.sent, dims=1)[1,:,:]
	tfr = sent_out + sent_in

	tot_tfr = zeros(Int, data.N, data.N)
	for i in 1:data.N, j in (i+1):data.N
		q = cumsum(results.sent[i,j,:] - results.sent[j,i,:])
		z = maximum(abs.(q))
		tot_tfr[i,j] = z
	end

	s = sum(results.shortage)
	s_null = sum(results.shortage_nosent)
	s_red = (s_null - s) / s_null

	total_nurse_days = sum(results.active_nosent)

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
		printmetric("Total nurses", sum(data.nurses))
		printmetric("Total nurse-days", sum(results.active_nosent))
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
		printmetric("Nurse shortage (without transfers)", s_null)
		printmetric("Nurse shortage (with transfers)", s)
		printmetric("Nurse shortage reduction", s_red, pct=true)
		println()
		printmetric("Percent shortage (without transfers)", s_null / total_nurse_days, pct=true)
		printmetric("Percent shortage (with transfers)", s / total_nurse_days, pct=true)
		printmetric("Shortage reduction", s_red, pct=true)
		println()
		printmetric("Number of hospital-days with a nurse shortage (without transfers)", sum(results.shortage_nosent .> 0))
		printmetric("Number of hospital-days with a nurse shortage (with transfers)", sum(results.shortage .> 0))
		printmetric("Percent of hospital-days with a nurse shortage (without transfers)", mean(results.shortage_nosent .> 0), pct=true)
		printmetric("Percent of hospital-days with a nurse shortage (with transfers)", mean(results.shortage .> 0), pct=true)
		println()
		printmetric("Maximum nurse shortage by hospital-day (without transfers)", maximum(results.shortage_nosent))
		printmetric("Maximum nurse shortage by hospital-day (with transfers)", maximum(results.shortage))
		println()
		printmetric("Maximum nurse load by hospital-day (without transfers)", maximum(skipbad(results.load_nosent)))
		printmetric("Maximum nurse load by hospital-day (with transfers)", maximum(skipbad(results.load)))
	end

	metrics = OrderedDict(
		"experiment date" => config.rundate,
		"experiment" => config.experiment,
		"region" => config.region,
		"allocation level" => config.alloc_level,
		"resource" => "nurses",
		"bed type" => bedtype,
		"start date" => data.start_date,
		"end date" => data.end_date,
		"number of days" => T,
		"number of hospitals" => N,
		"number of hospital-days" => N*T,
		"total nurses" => sum(data.nurses),
		"total nurse-days" => sum(results.active_nosent),
		"total nurses transferred" => sum(results.sent),
		"max sent by hospital-day" => maximum(results.sent),
		"max sent by day" => maximum(sum(results.sent, dims=[1,2])),
		"max sent by hospital" => maximum(sum(results.sent, dims=[2,3])),
		"number of transfers" => sum(results.sent .> 0),
		"proportion of hospital-days with a transfer" => mean(tfr .> 0),
		"proportion of days with a transfer" => mean(sum(tfr, dims=1) .> 0),
		"proportion of hospitals with a transfer" => mean(sum(tfr, dims=2) .> 0),
		"percent shortage (without transfers)" => (s_null / total_nurse_days),
		"percent shortage (with transfers)" => (s / total_nurse_days),
		"nurse shortage (without transfers)" => s_null,
		"nurse shortage (with transfers)" => s,
		"nurse shortage reduction" => s_red,
		"number of hospital-days with a nurse shortage (without transfers)" => sum(results.shortage_nosent .> 0),
		"number of hospital-days with a nurse shortage (with transfers)" => sum(results.shortage .> 0),
		"proportion of hospital-days with a nurse shortage (without transfers)" => mean(results.shortage_nosent .> 0),
		"proportion of hospital-days with a nurse shortage (with transfers)" => mean(results.shortage .> 0),
		"maximum nurse shortage by hospital-day (without transfers)" => maximum(results.shortage_nosent),
		"maximum nurse shortage by hospital-day (with transfers)" => maximum(results.shortage),
		"maximum nurse load by hospital-day (without transfers)" => maximum(skipbad(results.load_nosent)),
		"maximum nurse load by hospital-day (with transfers)" => maximum(skipbad(results.load)),
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
		"resource" => "nurses",
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

		"shortage" => roundmetric(s),
		"shortage reduction" => to_percent(s_red),

		"median non-zero shortage" => iszero(results.shortage) ? 0 : roundmetric(median(nonzero(results.shortage[:]))),
		"mean non-zero shortage" => iszero(results.shortage) ? 0 : roundmetric(mean(nonzero(results.shortage[:]))),
		"max non-zero shortage" => iszero(results.shortage) ? 0 : roundmetric(maximum(nonzero(results.shortage[:]))),

		"median load" => to_percent(median(skipbad(results.load[:]))),
		"mean load" => to_percent(mean(skipbad(results.load[:]))),
		"max load" => to_percent(maximum(skipbad(results.load[:]))),

		"number of hospital-days with a shortage" => roundmetric(sum(results.shortage .> 0)),
		"percent of hospital-days with a shortage" => to_percent(mean(results.shortage .> 0)),

		# transfer metrics

		"total nurse transfers" => roundmetric(sum(results.sent)),
		"percent of nurses transferred" => to_percent(sum(tot_tfr) / sum(data.nurses)),

		"median non-zero transfer" => iszero(results.sent) ? 0 : roundmetric(median(nonzero(results.sent[:]))),
		"mean non-zero transfer" => iszero(results.sent) ? 0 : roundmetric(mean(nonzero(results.sent[:]))),
		"max non-zero transfer" => iszero(results.sent) ? 0 : roundmetric(maximum(nonzero(results.sent[:]))),

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

function plot_shortage_distribution(config, data, results; bedtype=:all, display=true, save=true)
	colors = Scale.default_discrete_colors(4)[[1,3]]
	binwidth = 5
	shortage_nonzero = filter(x -> x > 0, results.shortage)
	shortage_nosent_nonzero = filter(x -> x > 0, results.shortage_nosent)
	nbins_sent = !isempty(shortage_nonzero) ? ceil(maximum(shortage_nonzero) / binwidth) : 1
	nbins_nosent = !isempty(shortage_nosent_nonzero) ? ceil(maximum(shortage_nosent_nonzero) / binwidth) : 1
	shortage_dist_plot = plot(
		layer(
			x=shortage_nonzero,
			Geom.histogram(
				bincount=nbins_sent,
				limits=(min=0, max=nbins_sent*binwidth)
			),
			style(default_color=colors[1], alphas=[0.7])
		),
		layer(
			x=shortage_nosent_nonzero,
			Geom.histogram(
				bincount=nbins_nosent,
				limits=(min=0, max=nbins_nosent*binwidth)
			),
			style(default_color=colors[2], alphas=[0.7])
		),
		Guide.title("Distribution of (Non-Zero) Shortage by Hospital-Day"),
		Guide.xlabel("Shortage"),
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
		shortage_dist_plot |> SVG(23cm, 10cm)
	end
	if save
		save_all(shortage_dist_plot, 23cm, 10cm, "shortage_dist"*bedtype_ext, config)
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
		plot_title = "Total COVID ICU Nurses in $(config.region)"
		fn = "active_total_icu"
	elseif bedtype == :ward
		plot_title = "Total COVID Ward Nurses in $(config.region)"
		fn = "active_total_ward"
	else
		plot_title = "Total COVID Nurses in $(config.region)"
		fn = "active_total"
	end

	overall_plt = Gadfly.plot(
		layer(
			x = data.date_range,
			y = sum(data.demand, dims=1)[:],
			Geom.point, Geom.line,
			style(default_color=colorant"red"),
		),
		layer(
			x = data.date_range,
			y = fill(sum(data.nurses), data.T),
			Geom.point, Geom.line,
		),
		Guide.xlabel("Date"), Guide.ylabel("Nurses"),
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

	key_labels = ["COVID Nurses", "COVID Nurse Demand"]
	key_colors = [Scale.default_discrete_colors(1)[1], colorant"red"]
	key_h = 0.5cm
	key = make_horizontal_key(key_labels, key_colors, height=key_h)

	overall_plt = compose(
		context(0,0,1,1),
		compose(
			context(0, 0, 1w, 1h - key_h),
			render(overall_plt),
		),
		compose(
			context(0, 1h - key_h, 1w, key_h),
			key,
		),
	)

	overall_plt = fill_background(overall_plt)

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
			layer(x=data.date_range, y=data.demand[h_idx,:], Geom.point, Geom.line, style(default_color=colorant"red")),
			Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
			Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
			Coord.Cartesian(ymin=0),
			Guide.xlabel("Date"), Guide.ylabel("Nurses"),
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
	active_key = make_horizontal_key(["Nurses (Without Transfers)", "Nurses (With Transfers)", "Nurse Demand"], active_key_colors, height=0.5cm)

	if bedtype == :all
		plot_title = "COVID Nurses With and Without Transfers"
	elseif bedtype == :icu
		plot_title = "COVID Nurses in the ICU With and Without Transfers"
	elseif bedtype == :ward
		plot_title = "COVID Nurses in Wards With and Without Transfers"
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

function plot_demand(config, data; add_title=true, columns=3, display=true, save=true, subset=nothing, bedtype=:all)
	if subset == nothing
		locs = enumerate(data.locations)
	else
		ind = [findfirst(==(h), data.locations) for h in subset]
		locs = zip(ind, subset)
	end

	active_plot_list = []
	for (h_idx, h_name) in locs
		plot_title = length(h_name) > 43 ? strip(h_name[1:40])*"..." : h_name
		plt = plot(
			layer(x=data.date_range, y=data.demand[h_idx,:], Geom.point, Geom.line, style(default_color=colorant"red")),
			Guide.xticks(ticks=DateTime(data.start_date):Day(14):DateTime(data.end_date)),
			Scale.x_continuous(labels=x -> Dates.format(x, dateformat"mm-dd")),
			Coord.Cartesian(ymin=0),
			Guide.xlabel("Date"), Guide.ylabel("Nurses"),
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

	if bedtype == :all
		plot_title = "COVID Nurse Demand"
	elseif bedtype == :icu
		plot_title = "COVID Nurse Demand in the ICU"
	elseif bedtype == :ward
		plot_title = "COVID Nurse Demand in Wards"
	end

	active_plots, dy = add_title ? add_title_top(plot_title, active_plots) : (active_plots, 0cm)
	active_plots = fill_background(active_plots)

	active_plots_w = active_plots_w
	active_plots_h = active_plots_h + dy

	if display
		active_plots |> SVG(active_plots_w, active_plots_h)
	end

	if save
		out_name = isnothing(subset) ? "demand" : "demand_subset"
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
		Guide.xlabel("Date"), Guide.ylabel("Nurse Load"),
		Guide.title("COVID Nurse Load in $(config.region) - Without Transfers"),
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
		Guide.xlabel("Date"), Guide.ylabel("Nurse Load"),
		Guide.title("COVID Nurse Load in $(config.region) - With Transfers"),
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
		Guide.xlabel("Date"), Guide.ylabel("Nurse Transfers"),
		Guide.title("Total Nurse Transfers"),
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
			Guide.xlabel("Date"), Guide.ylabel("Nurses"),
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
	transfer_plot, dy2 = add_title ? add_title_top("Nurses Sent and Received by Hospital", transfer_plot) : (transfer_plot, 0cm)
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

function plot_patients_per_nurse(config, combined_data; display=true, save=true)
	nbins = 16
	plt1 = plot(
		combined_data, x=:patients_per_nurse_icu,
		Geom.histogram(bincount=nbins),
		Guide.xlabel("Patient to Nurse Ratio"),
		Guide.ylabel("Frequency"),
		Guide.title("Patients per Nurse Distribution - ICU"),
		Coord.cartesian(xmin=0, xmax=4),
		style(
			minor_label_font_size=16px,
			major_label_font_size=18px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			background_color=colorant"white",
		),
	)
	plt2 = plot(
		combined_data, x=:patients_per_nurse_ward,
		Geom.histogram(bincount=nbins),
		Guide.xlabel("Patient to Nurse Ratio"),
		Guide.ylabel("Frequency"),
		Guide.title("Patients per Nurse Distribution - Ward"),
		Coord.cartesian(xmin=0, xmax=8),
		style(
			minor_label_font_size=16px,
			major_label_font_size=18px,
			minor_label_font="CMU Serif",
			major_label_font="CMU Serif",
			background_color=colorant"white",
		),
	)
	plt = hstack(plt1, plt2)

	if display
		plt |> SVG(28cm, 8cm)
	end
	if save
		config = merge(config, (experiment = "data",))
		save_all(plt, 28cm, 8cm, "patients_per_nurse", config)
	end
end;

end;
