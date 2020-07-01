module PatientNurseAllocation

using JuMP
using Gurobi

using LinearAlgebra
using MathOptInterface

export patient_nurse_allocation, patient_nurse_block_allocation


function patient_nurse_allocation(
		beds::Array{<:Real,1},
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		initial_nurses::Array{<:Real,1},
		adj_matrix::BitArray{2};
		hospitalized_days::Int=8,
		nurse_days_per_patient_day::Real=2.0,
		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		smoothness_penalty::Real=0,
		setup_cost::Real=0,
		sent_penalty::Real=0,
		balancing_thresh_patients::Real=1.0,
		balancing_penalty_patients::Real=0,
		nurse_target_load::Real=1.25,
		nurse_target_load_gap::Real=0.25,
		nurse_load_penalty::Real=0,
		disallow_nurse_shortage_sent::Bool=false,
		disallow_nurse_shortage_newpatients::Bool=false,
		severity_weighting::Bool=false,
		verbose::Bool=false,
)
	N, T = size(admitted_patients)
	@assert(size(initial_patients, 1) == N)
	@assert(size(beds, 1) == N)
	@assert(size(initial_nurses, 1) == N)
	@assert(size(adj_matrix) == (N,N))
	@assert(size(discharged_patients) == (N, T))

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	@variable(model, sentpatients[1:N,1:N,1:T])
	@variable(model, sentnurses[1:N,1:N,1:T])

	@variable(model, obj_dummy_patients[1:N,1:T] >= 0)
	@variable(model, obj_dummy_nurses[1:N,1:T] >= 0)

	# enforce minimum transfer amount if enabled
	if min_send_amt <= 0
		@constraint(model, sentpatients .>= 0)
		@constraint(model, sentnurses .>= 0)
	else
		@constraint(model, [i=1:N,j=1:N,t=1:T], sentpatients[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
		@constraint(model, [i=1:N,j=1:N,t=1:T], sentnurses[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
	end

	objective = @expression(model, 100.0 * sum(obj_dummy_nurses))

	active_patients_null = [(
			initial_patients[i]
			- sum(discharged_patients[i,1:t])
			+ sum(admitted_patients[i,max(1,t-hospitalized_days+1):t])
		) for i in 1:N, t in 1:T
	]

	if severity_weighting
		max_load_null = [maximum(active_patients_null[i,:] / beds[i]) for i in 1:N]
		severity_weight = [max_load_null[i] > 1 ? 1.0 : 10.0 for i in 1:N]

		add_to_expression!(objective, dot(sum(obj_dummy_patients, dims=2), severity_weight))
	else
		add_to_expression!(objective, sum(obj_dummy_patients))
	end

	# penalize total sent if enabled
	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*sum(sentpatients))
		add_to_expression!(objective, sent_penalty*sum(sentnurses))
	end

	# penalize non-smoothness in sent patients if enabled
	if smoothness_penalty > 0
		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  (sentpatients[:,:,t] - sentpatients[:,:,t+1]) .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -(sentpatients[:,:,t] - sentpatients[:,:,t+1]) .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sentpatients[:,:,1]))
	end

	# add setup costs if enabled
	if setup_cost > 0
		@variable(model, setup_dummy_patients[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy_patients[i,j], sum(sentpatients[i,j,:])+sum(sentpatients[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy_patients))

		@variable(model, setup_dummy_nurses[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy_nurses[i,j], sum(sentnurses[i,j,:])+sum(sentnurses[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy_nurses))
	end

	# only send patients between connected locations
	for i = 1:N
		@constraint(model, sentnurses[i,i,:] .== 0)
		for j = 1:N
			if ~adj_matrix[i,j]
				@constraint(model, sentpatients[i,j,:] .== 0)
				# @constraint(model, sentnurses[i,j,:] .== 0)
			end
		end
	end

	# enforce a minimum time between sending and receiving
	if sendreceive_switch_time > 0
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentpatients[:,i,t]), sum(sentpatients[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0]))
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentpatients[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sentpatients[i,:,t])] in MOI.SOS1([1.0, 1.0]))
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentnurses[:,i,t]), sum(sentnurses[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0]))
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentnurses[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sentnurses[i,:,t])] in MOI.SOS1([1.0, 1.0]))
	end

	# send new patients only
	@constraint(model, [t=1:T], sum(sentpatients[:,:,t], dims=2) .<= admitted_patients[:,t])

	# expression for the number of active patients
	@expression(model, active_patients[i=1:N,t=0:T],
		initial_patients[i]
		- sum(discharged_patients[i,1:t])
		+ sum(admitted_patients[i,max(1,t-hospitalized_days+1):t])
		- sum(sentpatients[i,:,max(1,t-hospitalized_days+1):t])
		+ sum(sentpatients[:,i,max(1,t-hospitalized_days+1):t])
	)

	# ensure the number of active patients is non-negative
	@constraint(model, [i=1:N,t=1:T], active_patients[i,t] >= 0)

	# load balancing for patients
	if balancing_penalty_patients > 0
		@variable(model, balancing_dummy_patients[1:N,1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], balancing_dummy_patients[i,t] >= (active_patients[i,t] / beds[i]) - balancing_thresh_patients)
		add_to_expression!(objective, balancing_penalty_patients * sum(balancing_dummy_patients))
	end

	# objective - patients
	@expression(model, patient_overflow[i=1:N,t=1:T], active_patients[i,t] + sum(sentpatients[i,:,t]) - beds[i])
	@constraint(model, [i=1:N,t=1:T], obj_dummy_patients[i,t] >= patient_overflow[i,t])

	# @constraint(model, sentpatients .== 0)

	# compute active nurses
	@expression(model, active_nurses[i=1:N,t=0:T],
		initial_nurses[i]
		- sum(sentnurses[i,:,1:t])
		+ sum(sentnurses[:,i,1:t])
	)

	# compute nurse demand
	@expression(model, nurse_demand[i=1:N,t=1:T], active_patients[i,t] * nurse_days_per_patient_day)

	# sent nurses ≦ active nurses
	@constraint(model, [i=1:N,t=1:T], sum(sentnurses[i,:,t]) <= active_nurses[i,t-1])

	# active nurses ≧ 1/2 initial nurses
	@constraint(model, [i=1:N,t=1:T], active_nurses[i,t] >= 0.5 * initial_nurses[i])

	# nurses objective
	@constraint(model, [i=1:N,t=1:T], obj_dummy_nurses[i,t] >= nurse_demand[i,t] - active_nurses[i,t])

	if disallow_nurse_shortage_sent
		# m = 1e-5
		# @variable(model, has_nurse_shortage[i=1:N,t=1:T], Bin)
		# @constraint(model, [i=1:N,t=1:T],     m*(nurse_demand[i,t] - active_nurses[i,t]) <= has_nurse_shortage[i,t])
		# @constraint(model, [i=1:N,t=1:T], 1 + m*(nurse_demand[i,t] - active_nurses[i,t]) >= has_nurse_shortage[i,t])
		# @constraint(model, [i=1:N,t=1:T], has_nurse_shortage[i,t] => {active_nurses[i,t] >= initial_nurses[i]})
		nurse_demand_null = active_patients_null .* nurse_days_per_patient_day
		for i in 1:N, t in 1:T
			if nurse_demand_null[i,t] >= initial_nurses[i]
				@constraint(model, sum(sentnurses[:,i,1:t]) >= sum(sentnurses[i,:,1:t]))
			end
		end
	end

	if disallow_nurse_shortage_newpatients
		m = 1e-5
		ts(t) = max(1,t-hospitalized_days+1)
		@variable(model, has_outside_patients[i=1:N,t=1:T], Bin)
		@constraint(model, [i=1:N,t=1:T],     m*(sum(sentpatients[:,i,ts(t):t]) - sum(sentpatients[i,:,ts(t):t])) <= has_outside_patients[i,t])
		@constraint(model, [i=1:N,t=1:T], 1 + m*(sum(sentpatients[:,i,ts(t):t]) - sum(sentpatients[i,:,ts(t):t])) >= has_outside_patients[i,t])
		@constraint(model, [i=1:N,t=1:T], has_outside_patients[i,t] => {active_nurses[i,t] >= nurse_demand[i,t]})
	end

	# nurse load
	if nurse_load_penalty > 0
		@variable(model, load_dummy_nurses_abs[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T],  (active_nurses[i,t] - nurse_target_load*nurse_demand[i,t]) <= load_dummy_nurses_abs[i,t])
		@constraint(model, [i=1:N,t=1:T], -(active_nurses[i,t] - nurse_target_load*nurse_demand[i,t]) <= load_dummy_nurses_abs[i,t])

		@variable(model, load_dummy_nurses[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], load_dummy_nurses[i,t] >= load_dummy_nurses_abs[i,t] - nurse_target_load_gap*nurse_demand[i,t])
		add_to_expression!(objective, nurse_load_penalty * sum(load_dummy_nurses))
	end

	@objective(model, Min, objective)

	optimize!(model)
	return model
end;

function patient_nurse_block_allocation(
		beds::Dict{Symbol,Array{Float32,1}},
		patient_blocks::Array,
		nurses::Array{Float32,1},
		adj_matrix::BitArray{2};
		nurse_hrs_per_week::Real=36,
		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		smoothness_penalty::Real=0,
		setup_cost::Real=0,
		sent_penalty::Real=0,
		balancing_thresh::Real=1.0,
		balancing_penalty::Real=0,
		nurse_target_load::Real=1.25,
		nurse_dead_zone::Real=0.25,
		nurse_load_penalty::Real=10.0,
		verbose::Bool=false,
)
	N, T = size(patient_blocks[1].admitted)
	G = length(patient_blocks)
	B = length(beds)

	bed_types = collect(keys(beds))

	nurse_hrs_per_day = nurse_hrs_per_week / 7
	nurses_days_per_day = 24 / nurse_hrs_per_day
	nurse_days_per_patient_day(g) = Float32(nurses_days_per_day / patient_blocks[g].patients_per_nurse)

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	@variable(model, sent[1:G,1:N,1:N,1:T])
	@variable(model, obj_dummy[bed_types,1:N,1:T])

	@variable(model, sentnurses[1:N,1:N,1:T] >= 0)
	@variable(model, obj_dummy_nurses[1:N,1:T] >= 0)

	# enforce minimum transfer amount if enabled
	if min_send_amt <= 0
		@constraint(model, sent .>= 0)
		@constraint(model, sentnurses .>= 0)
	else
		@constraint(model, [i=1:N,j=1:N,t=1:T], sent[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
		@constraint(model, [i=1:N,j=1:N,t=1:T], sentnurses[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
	end

	objective = @expression(model, sum(obj_dummy) + sum(obj_dummy_nurses))

	# penalize total sent if enabled
	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*sum(sent))
		add_to_expression!(objective, sent_penalty*sum(sentnurses))
	end

	# penalize non-smoothness in sent patients if enabled
	if smoothness_penalty > 0
		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  sum(sent[:,:,:,t] - sent[:,:,:,t+1], dims=1) .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -sum(sent[:,:,:,t] - sent[:,:,:,t+1], dims=1) .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sent[:,:,:,1]))
	end

	# add setup costs if enabled
	if setup_cost > 0
		@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[:,i,j,:])+sum(sent[:,j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy))
	end

	# only send patients between connected locations
	for i = 1:N
		for j = 1:N
			if ~adj_matrix[i,j]
				@constraint(model, sent[:,i,j,:] .== 0)
				@constraint(model, sentnurses[i,j,:] .== 0)
			end
		end
	end

	# enforce a minimum time between sending and receiving
	if sendreceive_switch_time > 0
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,:,i,t]), sum(sent[:,i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0])
		)
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,:,i,t:min(t+sendreceive_switch_time,T)]), sum(sent[:,i,:,t])] in MOI.SOS1([1.0, 1.0])
		)
	end

	ts(t,g) = max(1,t-patient_blocks[g].hospitalized_days+1)

	@expression(model, transfer[g1=1:G,g2=1:G,i=1:N,t=1:T],
		((t > patient_blocks[g1].hospitalized_days) && (patient_blocks[g1].to == patient_blocks[g2].id)) ?
			patient_blocks[g1].admitted[i,t-patient_blocks[g1].hospitalized_days] : 0
	)

	# expression for the number of active patients
	@expression(model, active_patients[g=1:G,i=1:N,t=0:T],
		patient_blocks[g].initial[i]
		- sum(patient_blocks[g].discharged[i,1:t])
		+ sum(patient_blocks[g].admitted[i,ts(t,g):t])
		- sum(sent[g,i,:,ts(t,g):t])
		+ sum(sent[g,:,i,ts(t,g):t])
		+ sum(transfer[:,g,i,ts(t,g):t])
	)

	# expression for the patient overflow
	bed_groups = Dict(b => [i for (i,g) in enumerate(patient_blocks) if g.bed_type == b] for b in bed_types)
	@expression(model, overflow[b in bed_types,i=1:N,t=1:T],
		sum(active_patients[g,i,t] + sum(sent[g,i,:,t]) for g in bed_groups[b]) - beds[b][i]
	)

	# send new patients only
	@constraint(model, [g=1:G,i=1:N,t=1:T], sum(sent[g,i,:,t]) <= patient_blocks[g].admitted[i,t])

	# sent nurses <= active nurses
	@constraint(model, [i=1:N,t=1:T],
		sum(sentnurses[i,:,t]) <=
			nurses[i]
			- sum(sentnurses[i,:,1:t-1])
			+ sum(sentnurses[:,i,1:t-1])
	)

	# ensure the number of active patients is non-negative
	@constraint(model, [g=1:G,i=1:N,t=1:T], active_patients[g,i,t] >= 0)

	# load balancing
	if balancing_penalty > 0
		@variable(model, balancing_dummy[bed_types,1:N,1:T] >= 0)
		@constraint(model, [b in bed_types,i=1:N,t=1:T],
			balancing_dummy[b,i,t] >= (sum(active_patients[g,i,t] for g in bed_groups[b]) / beds[b][i]) - balancing_thresh)
		add_to_expression!(objective, balancing_penalty * sum(balancing_dummy))
	end

	# objective - patients
	@constraint(model, [b in bed_types,i=1:N,t=1:T], obj_dummy[b,i,t] >= overflow[b,i,t])

	@expression(model, active_nurses[i=1:N,t=1:T],
		nurses[i]
		- sum(sentnurses[i,:,1:t])
		+ sum(sentnurses[:,i,1:t-1])
	)
	@expression(model, total_nurse_demand[i=1:N,t=1:T],
		sum(active_patients[g,i,t] * nurse_days_per_patient_day(g) for g in 1:G)
	)
	@expression(model, nurse_demand[g=1:G,i=1:N,t=1:T],
		active_patients[g,i,t] * nurse_days_per_patient_day(g)
	)

	# objective - nurses
	@constraint(model, [i=1:N,t=1:T], obj_dummy_nurses[i,t] >= total_nurse_demand[i,t] - active_nurses[i,t])

	if nurse_load_penalty > 0
		@variable(model, nurse_load_dummy_abs[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T],  (active_nurses[i,t] - nurse_target_load*total_nurse_demand[i,t]) <= nurse_load_dummy_abs[i,t])
		@constraint(model, [i=1:N,t=1:T], -(active_nurses[i,t] - nurse_target_load*total_nurse_demand[i,t]) <= nurse_load_dummy_abs[i,t])

		@variable(model, nurse_load_dummy[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], nurse_load_dummy[i,t] >= nurse_load_dummy_abs[i,t] - nurse_dead_zone*total_nurse_demand[i,t])
		add_to_expression!(objective, nurse_load_penalty * sum(nurse_load_dummy))
	end

	@objective(model, Min, objective)
	optimize!(model)

	return model
end;

end;
