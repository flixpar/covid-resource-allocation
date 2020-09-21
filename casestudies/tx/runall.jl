##################################################
##### COVID Patient Re-Distribution -- Texas #####
##################################################
printstyled("COVID Patient Re-Distribution -- Texas\n", bold=true)


#############
### Setup ###
#############
println("Setup")

## Includes ##
println("Includes")
using Dates
using Distributions
using JuMP
using Serialization
ENV["COLUMNS"] = 1000;

projectbasepath = "../../";
include(normpath(projectbasepath, "src/models/PatientAllocation.jl"));
include(normpath(projectbasepath, "src/util/PatientAllocationFigures.jl"));

include("tx_data.jl");


## Configuation ##
println("Config")
results_basepath = abspath(joinpath(projectbasepath, "results/"));
paperfigures_basepath = abspath(joinpath(projectbasepath, "figures/"));

if !isdir(results_basepath) mkpath(results_basepath) end;
if !isdir(paperfigures_basepath) mkpath(paperfigures_basepath) end;

shared_config = (
    rundate = today(),
    region = "Texas",
    region_abbrev = "tx",
    alloc_level = "tsa",
    results_basepath = results_basepath,
    paperfigures_basepath = paperfigures_basepath,
);

saves_path = "$(shared_config.results_basepath)/$(shared_config.region_abbrev)/$(shared_config.rundate)/saves/"
if !isdir(saves_path) mkpath(saves_path) end;

start_date = Date(2020, 6, 15);
end_date   = Date(2020, 8, 15);
date_range = collect(start_date : Day(1) : end_date);
T = length(date_range);

locations_limit = nothing;
focus_locations = [
    "Austin",
    "Dallas/Ft. Worth",
    "El Paso",
    "Galveston",
    "Houston",
    "San Antonio",
];

pct_beds_available = (
    icu = 0.5,
    ward = 0.35,
    allpat = 0.4,
);
los_dist = (
    icu = Weibull(1.58, 13.32),
    ward = Weibull(1.38, 12.88),
    allpat = Weibull(1.38, 12.88),
);

## Helper Functions ##
println("Helper functions")

function save_results(experiment_name, _sent)
    serialize(joinpath(saves_path, "sent_$(experiment_name).jldata"), _sent)
    return
end;

function print_solve_metrics(_model)
    println("termination status:       ", termination_status(_model))
    println("solve time:               ", round(solve_time(_model), digits=3), "s")
    println("objective function value: ", round(objective_value(_model), digits=3))
    return
end;

function make_data_figures(_data)
    _config = merge(shared_config, (experiment = "data",))
    _sent = zeros(Int, _data.N, _data.N, _data.T)
    _results = PatientAllocationFigures.compute_results(_config, _data, _sent, use_rounding=true)

    PatientAllocationFigures.plot_active_total(_config, _data, _results, display=false)

    return
end;

function make_all_figures_base(experiment_name, _data, _sent)
    _config = merge(shared_config, (experiment = experiment_name,))

    _results = PatientAllocationFigures.compute_results(_config, _data, _sent, use_rounding=true)

    PatientAllocationFigures.plot_metrics(_config, _data, _results, display=false)

    PatientAllocationFigures.plot_estimates_total(_config, _data, display=false)

    PatientAllocationFigures.plot_overflow_distribution(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_load_distribution(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_maxload_distribution(_config, _data, _results, display=false)

    PatientAllocationFigures.plot_sent_total(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_transfers(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_transfers(_config, _data, _results, display=false, subset=_data.focus_locations)

    PatientAllocationFigures.plot_load(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_load(_config, _data, _results, display=false, subset=_data.focus_locations)

    PatientAllocationFigures.plot_active(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_active(_config, _data, _results, display=false, subset=_data.focus_locations)

    PatientAllocationFigures.plot_figures_list(_config)

    return _results
end;

function make_all_figures_block(experiment_name, _data, _sent)
    _config = merge(shared_config, (experiment = experiment_name,))

    _results = PatientAllocationFigures.compute_results_block(_config, _data, _sent, use_rounding=true);

    PatientAllocationFigures.plot_active_total(_config, _data.icu, _results.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_active_total(_config, _data.ward, _results.ward, bedtype=:ward, display=false);

    PatientAllocationFigures.plot_metrics(_config, _data.icu, _results.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_metrics(_config, _data.ward, _results.ward, bedtype=:ward, display=false);

    PatientAllocationFigures.plot_estimates_total(_config, _data.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_estimates_total(_config, _data.ward, bedtype=:ward, display=false);

    PatientAllocationFigures.plot_overflow_distribution(_config, _data.icu, _results.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_load_distribution(_config, _data.icu, _results.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_maxload_distribution(_config, _data.icu, _results.icu, bedtype=:icu, display=false);

    PatientAllocationFigures.plot_overflow_distribution(_config, _data.ward, _results.ward, bedtype=:ward, display=false);
    PatientAllocationFigures.plot_load_distribution(_config, _data.ward, _results.ward, bedtype=:ward, display=false);
    PatientAllocationFigures.plot_maxload_distribution(_config, _data.ward, _results.ward, bedtype=:ward, display=false);

    PatientAllocationFigures.plot_sent_total(_config, _data.icu, _results.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_transfers(_config, _data.icu, _results.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_transfers(_config, _data.icu, _results.icu, bedtype=:icu, display=false, subset=_data.focus_locations);

    PatientAllocationFigures.plot_sent_total(_config, _data.ward, _results.ward, bedtype=:ward, display=false);
    PatientAllocationFigures.plot_transfers(_config, _data.ward, _results.ward, bedtype=:ward, display=false);
    PatientAllocationFigures.plot_transfers(_config, _data.ward, _results.ward, bedtype=:ward, display=false, subset=_data.focus_locations);

    PatientAllocationFigures.plot_load(_config, _data.icu, _results.icu, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_load(_config, _data.icu, _results.icu, bedtype=:icu, display=false, subset=_data.focus_locations);

    PatientAllocationFigures.plot_load(_config, _data.ward, _results.ward, bedtype=:ward, display=false);
    PatientAllocationFigures.plot_load(_config, _data.ward, _results.ward, bedtype=:ward, display=false, subset=_data.focus_locations);

    PatientAllocationFigures.plot_active(_config, _data.icu, _results.icu, add_title=true, bedtype=:icu, display=false);
    PatientAllocationFigures.plot_active(_config, _data.icu, _results.icu, add_title=true, bedtype=:icu, display=false, subset=_data.focus_locations);

    PatientAllocationFigures.plot_active(_config, _data.ward, _results.ward, add_title=true, bedtype=:ward, display=false);
    PatientAllocationFigures.plot_active(_config, _data.ward, _results.ward, add_title=true, bedtype=:ward, display=false, subset=_data.focus_locations);

    return _results
end;

function make_all_figures_robust(experiment_name, _data, _sent, _sent_base)
    _config = merge(shared_config, (experiment = experiment_name,))

    _results = PatientAllocationFigures.compute_results(_config, _data, _sent, use_rounding=true)
    _results = merge(_results, (sent_robust = _sent, sent_nonrobust = _sent_base));

    PatientAllocationFigures.plot_active_robust(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_active_robust(_config, _data, _results, display=false, subset=_data.focus_locations)

    PatientAllocationFigures.plot_active_samples_notransfers(_config, _data, _results, display=false)
    PatientAllocationFigures.plot_active_samples_notransfers(_config, _data, _results, display=false, subset=_data.focus_locations)

    PatientAllocationFigures.plot_robust_overflow_distribution(_config, _data, _results, debug=false, display=false)

    PatientAllocationFigures.plot_figures_list(_config)

    return _results
end;


############
### Data ###
############
println("Data")

println("Load data")
data = TexasData.load_data_tx(
    date_range,
    los_dist, pct_beds_available,
    focus_locations, locations_limit,
    use_rounding=true,
);

println("Save data")
serialize(joinpath(saves_path, "data.jldata"), data);
make_data_figures(data.icu);


##################
### Run models ###
##################
println("Run models")

## No Transfer Model ##
println("No Transfer Model")
sent_null = zeros(Float64, data.N, data.N, data.T);
results_null = make_all_figures_base("no_transfers", data.icu, sent_null);
println("===========================================================\n")

## Base Model ##
println("Base Model")
model_base = PatientAllocation.patient_allocation(
    data.icu.beds,
    data.icu.initial,
    data.icu.discharged,
    data.icu.admitted,
    data.icu.adj,
    los=data.icu.los_dist,
    verbose=true,
)
sent_base = value.(model_base[:sent])
save_results("base", sent_base)
print_solve_metrics(model_base)
results_base = make_all_figures_base("base", data.icu, sent_base);
println("===========================================================\n")

## Operational Model ##
println("Operational Model")
model_operational = PatientAllocation.patient_allocation(
    data.icu.beds,
    data.icu.initial,
    data.icu.discharged,
    data.icu.admitted,
    data.icu.adj,
    los=data.icu.los_dist,

    smoothness_penalty = 0.01,
    sent_penalty = 0.01,
    no_artificial_overflow = true,
    no_worse_overflow = true,
    capacity_cushion = 0.05,

    verbose=true,
)
sent_operational = value.(model_operational[:sent])
save_results("operational", sent_operational)
print_solve_metrics(model_operational)
results_operational = make_all_figures_base("operational", data.icu, sent_operational);
println("===========================================================\n")

## Robust Base Model ##
println("Robust Base Model")
model_robust = PatientAllocation.patient_allocation_robust(
    data.icu.beds,
    data.icu.initial,
    data.icu.discharged,
    data.icu.admitted,
    data.icu.admitted_uncertainty,
    data.icu.adj,
    los=data.icu.los_dist,

    Γ = 7,

    verbose=true,
)
sent_robust = value.(model_robust[:sent])
save_results("robust_base", sent_robust)
print_solve_metrics(model_robust)
make_all_figures_base("robust_base", data.icu, sent_robust);
results_robust = make_all_figures_robust("robust_base", data.icu, sent_robust, sent_base);
println("===========================================================\n")

## Robust Operational Model ##
println("Robust Operational Model")
model_robust_operational = PatientAllocation.patient_allocation_robust(
    data.icu.beds,
    data.icu.initial,
    data.icu.discharged,
    data.icu.admitted,
    data.icu.admitted_uncertainty,
    data.icu.adj,
    los=data.icu.los_dist,

    Γ = 7,

    smoothness_penalty = 0.01,
    sent_penalty = 0.01,
    no_artificial_overflow = true,
    no_worse_overflow = true,
    capacity_cushion = 0.05,

    verbose=true,
)
sent_robust_operational = value.(model_robust_operational[:sent])
save_results("robust_operational", sent_robust_operational)
print_solve_metrics(model_robust_operational)
make_all_figures_base("robust_operational", data.icu, sent_robust_operational);
results_robust_operational = make_all_figures_robust("robust_operational", data.icu, sent_robust_operational, sent_operational);
println("===========================================================\n")

## Block Base Model ##
println("Block Base Model")
model_block = PatientAllocation.patient_block_allocation(
    data.carepaths.beds,
    data.carepaths.initial,
    data.carepaths.discharged,
    data.carepaths.admitted,
    data.carepaths.los_bygroup,
    data.carepaths.adj,
    data.carepaths.group_transfer_graph,
    data.carepaths.bedtype_bygroup,
    verbose=true,
)
sent_block = value.(model_block[:sent])
save_results("block_base", sent_block)
print_solve_metrics(model_block)
results_block = make_all_figures_block("block_base", data.carepaths, sent_block);
println("===========================================================\n")

## Block Operational Model ##
println("Block Operational Model")
model_block_operational = PatientAllocation.patient_block_allocation(
    data.carepaths.beds,
    data.carepaths.initial,
    data.carepaths.discharged,
    data.carepaths.admitted,
    data.carepaths.los_bygroup,
    data.carepaths.adj,
    data.carepaths.group_transfer_graph,
    data.carepaths.bedtype_bygroup,

    smoothness_penalty = 0.01,
    sent_penalty = 0.01,
    no_artificial_overflow = true,
    no_worse_overflow = true,
    capacity_cushion = 0.05,

    verbose=true,
)
sent_block_operational = value.(model_block_operational[:sent])
save_results("block_operational", sent_block_operational)
print_solve_metrics(model_block_operational)
results_block_operational = make_all_figures_block("block_operational", data.carepaths, sent_block_operational);
println("===========================================================\n")

## Compare Results ##
println("Compare Results")
PatientAllocationFigures.plot_metrics_compare(shared_config, data.icu, [
    "no_transfer" => results_null,
    "base" => results_base,
    "operational" => results_operational,
    "robust_base" => results_robust,
    "robust_operational" => results_robust_operational,
], display_table=true)


##################
#### Complete ####
##################
println("\nComplete")
