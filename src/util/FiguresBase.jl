module FiguresBase

using DataFrames
using Gadfly
using Compose
import Cairo, Fontconfig

export plot_grid
export add_title_top, add_key_bottom, add_key_top, fill_background
export make_horizontal_key, make_key
export metrics_totextable
export get_output_path, get_output_folder, save_all
export skipbad, nonzero, printmetric


function plot_grid(_plots, ncols; plot_width=10, plot_height=8, return_plot=false, display=true)
	plots = convert(Array{Union{Plot,Compose.Context},1}, _plots)

	if (length(plots) % ncols) != 0
		nblank = ncols - (length(plots) % ncols)
		for i in 1:nblank
			push!(plots, Compose.context())
		end
	end

	nrows = Int(length(plots) / ncols)
	plots_grid = permutedims(reshape(plots, (ncols, nrows)), (2,1))

	plt_h = plot_height*cm
	plt_w = plot_width*cm
	plt = gridstack(plots_grid)

	if display
		plt |> SVG(ncols*plt_w, nrows*plt_h)
	end

	if return_plot
		return plt, ncols*plt_w, nrows*plt_h
	end

	return
end;

function add_title_top(title, _plot)
	_font = "CMU Serif"
	_fontsize = 22px
	_topmargin = 10px
	_bottommargin = 15px
	_totalsize = _topmargin + _fontsize + _bottommargin

	plot = compose(
		context(0, 0, 1w, 1h),
		compose(context(0, 0, 1w, 1h), text(0.5w, _topmargin, title, hcenter, vtop), fontsize(_fontsize), font(_font)),
		compose(context(0, _totalsize, 1w, 1h - _totalsize), _plot),
	)
	return plot, _totalsize
end;

function add_key_top(key, _plot; topmargin=0px, bottommargin=10px, keyheight=22px)
	totalsize = topmargin + keyheight + bottommargin
	plot = compose(
		context(0, 0, 1w, 1h),
		compose(context(0, topmargin, 1w, keyheight), key),
		compose(context(0, totalsize, 1w, 1h - totalsize), _plot),
	)
	return plot, totalsize
end;

function add_key_bottom(key, _plot; topmargin=15px, keyheight=22px)
	totalsize = topmargin + keyheight
	plot = compose(
		context(0, 0, 1w, 1h),
		compose(context(0, 0, 1w, 1h - totalsize), _plot),
		compose(context(0, 1h - keyheight, 1w, keyheight), key),
	)
	return plot, totalsize
end;

function fill_background(_plot)
	plot = compose(
		context(0, 0, 1, 1),
		_plot,
		compose(context(0, 0, 1w, 1h, order=-100), rectangle(), fill("white")),
	)
	return plot
end;

function make_horizontal_key(labels, colors; debug=false, height=22px)
	n = length(labels)

	key_labels = []

	_boxsize = height
	_fontsize = 0.8 * _boxsize
	_padding_between = 16px
	_padding_inside = 5px
	_font = "CMU Serif"

	ws = [_boxsize + _padding_inside + _padding_between + 1.1*text_extents("CMU Serif", _fontsize, l)[1][1] for l in labels]
	tot_w = sum(ws) - _padding_between
	pad_left = (1w - tot_w) / 2

	debug_box_outside = !debug ? nothing : compose(context(0, 0, 1, 1), rectangle(), fill(nothing), stroke("black"))

	x = pad_left
	for i in 1:n
		txt_ext = text_extents(_font, _fontsize, labels[i])[1]
		_w = _boxsize + _padding_inside + 1.1*txt_ext[1]
		width = _w + _padding_between
		debug_box_inside = !debug ? nothing : compose(context(0, 0, 1, 1), rectangle(0, 0, _w, 1), stroke("black"), fill(nothing))
		label = compose(
			context(x, 0, width, 1h),
			compose(
				context(0, 0, _boxsize, _boxsize),
				rectangle(),
				fill(colors[i]),
			),
			compose(
				context(0, 0, 1w, 1h),
				text(_boxsize + _padding_inside, _boxsize/2, labels[i], hleft, vcenter),
				fontsize(_fontsize),
				font(_font),
			),
			debug_box_inside,
		)
		push!(key_labels, label)
		x += width
	end

	key = compose(
		context(0, 0, 1, 1),
		key_labels...,
		debug_box_outside,
	)

	return key
end;

function make_key(labels, colors; debug=false, line_height=22px, ncolumns=3, shape=:box)
	n = length(labels)
	nrows = Int(ceil(n / ncolumns))

	_boxsize = line_height
	_fontsize = 0.8 * _boxsize
	_padding_between = 16px
	_padding_inside = 5px
	_padding_vertical = 3px
	_font = "CMU Serif"

	@assert shape == :box || shape == :circle
	_labelshape = shape == :box ? Compose.rectangle() : Compose.circle(0.5cx, 0.5cy, 0.4*_boxsize)

	ws = [_boxsize + _padding_inside + _padding_between + text_extents("CMU Serif", _fontsize, l)[1][1] for l in labels]
	col_width = maximum(ws)
	row_width = (ncolumns * col_width) - _padding_between
	pad_left = (1w - row_width) / 2

	debug_box_outside = !debug ? nothing : compose(context(0, 0, 1, 1), rectangle(), fill(nothing), stroke("black"))

	key_labels = []
	y = 0cm
	for j in 0:nrows-1
		x = pad_left
		for k in 1:ncolumns
			i = (ncolumns * j) + k
			if i > n
				continue
			end
			debug_box_inside = !debug ? nothing : compose(
				context(0, 0, 1, 1),
				rectangle(0, 0, ws[i] - _padding_between, _boxsize),
				stroke("black"),
				fill(nothing),
			)
			label = compose(
				context(x, y, ws[i], 1h),
				compose(
					context(0, 0, _boxsize, _boxsize),
					_labelshape,
					fill(colors[i]),
				),
				compose(
					context(0, 0, 1w, 1h),
					text(_boxsize + _padding_inside, _boxsize/2, labels[i], hleft, vcenter),
					fontsize(_fontsize),
					font(_font),
				),
				debug_box_inside,
			)
			push!(key_labels, label)
			x += col_width
		end
		y += _boxsize + _padding_vertical
	end

	height = y - _padding_vertical

	key = compose(
		context(0, 0, 1, 1),
		key_labels...,
		debug_box_outside,
	)

	return key, height
end;

function metrics_totextable(metrics::DataFrame; header::Bool=false)
	n_value_cols = ncol(metrics) - 1
	value_cols_header = repeat("r|", n_value_cols)[1:end-1]
	s = "\\begin{tabular}{l|$(value_cols_header)}\n"
	if (header) s *= "\\hline\\hline\n" end
	for (i,row) in enumerate(eachrow(metrics))
		r = ""
		for col in names(metrics)
			v = row[col]
			v = (v isa String && tryparse(Int, v) != nothing) ? parse(Int, v) : v
			v = (v isa String && tryparse(Float64, v) != nothing) ? parse(Float64, v) : v
			v = (v isa Float64 && round(Int, v) == v) ? round(Int, v) : v
			v = v isa Float64 ? round(v, digits=1) : v
			v = v isa Real ? "\$$(v)\$" : v
			v = v isa String ? titlecase(replace(v, "_" => " ")) : v
			v = (v isa String && v[end] == '%') ? "\$$(v[1:end-1])\\%\$" : v
			r *= "$(v) & "
		end
		r = r[1:end-3]
		s *= "$(r)\\\\\n"
		if (i == 1 && header) s *= "\\hline\n" end
	end
	s *= "\\end{tabular}"
	return s
end;

get_output_path(config, ext, fn) = "$(get_output_folder(config, ext))/$(fn).$(ext)"
function get_output_folder(config, ext)
	p = "$(config.results_basepath)/$(config.region_abbrev)/$(config.rundate)/$(config.experiment)/$(ext)/"
	if !isdir(p) mkpath(p) end
	return p
end;

get_figures_path(config, ext, fn) = "$(get_figures_folder(config))/$(fn).$(ext)"
function get_figures_folder(config)
	p = "$(config.paperfigures_basepath)/$(config.region_abbrev)/$(config.experiment)/"
	if !isdir(p) mkpath(p) end
	return p
end;

function save_all(figure, fig_w, fig_h, fn, config)
	figure |> PDF(get_output_path(config, "pdf", fn), fig_w, fig_h)
	figure |> PNG(get_output_path(config, "png", fn), fig_w, fig_h, dpi=300)
	figure |> PGF(get_output_path(config, "tex", fn), fig_w, fig_h, false, texfonts=true)
	figure |> SVG(get_output_path(config, "svg", fn), fig_w, fig_h)
	figure |>  PS(get_output_path(config, "eps", fn), fig_w, fig_h)
	figure |> PDF(get_figures_path(config, "pdf", fn), fig_w, fig_h)
	return
end;

skipbad(x) = filter(y -> !(ismissing(y) || isnothing(y) || isnan(y) || isinf(y)), x);
nonzero(x) = filter(!=(0), x)
printmetric(m, x; pct=false, int=false) = println("$(m): $(int ? x : round((pct ? x*100 : x),digits=2))" * (pct ? "%" : ""))

end;
