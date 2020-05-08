module PatientAllocation

using JuMP
using Gurobi

using LinearAlgebra
using MathOptInterface

export patient_allocation

function patient_allocation(
        beds::Array{Float32,1},
        initial_patients::Array{Float32,1},
        admitted_patients::Array{Float32,2},
        discharged_patients::Array{Float32,2},
        adj_matrix::BitArray{2};
        objective::Symbol=:overflow,
        hospitalized_days::Int=8,
        send_new_only::Bool=true,
        sendrecieve_switch_time::Int=0,
        min_send_amt::Real=0,
        smoothness_penalty::Real=0,
        setup_cost::Real=0,
        sent_penalty::Real=0,
        verbose::Bool=false,
)
    N, T = size(admitted_patients)
    @assert(size(initial_patients, 1) == N)
    @assert(size(beds, 1) == N)
    @assert(size(adj_matrix) == (N,N))
    @assert(size(discharged_patients) == (N, T))
    @assert(objective == :overflow)

    model = Model(Gurobi.Optimizer)
    if !verbose set_silent(model) end

    @variable(model, sent[1:N,1:N,1:T])
    @variable(model, obj_dummy[1:N,1:T] >= 0)

    if min_send_amt <= 0
        @constraint(model, sent .>= 0)
    else
        @constraint(model, [i=1:N,j=1:N,t=1:T], sent[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
    end

    objective = @expression(model, sum(obj_dummy))
    if sent_penalty > 0
        add_to_expression!(objective, sent_penalty*sum(sent))
    end
    if smoothness_penalty > 0
        @variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
        @constraint(model, [t=1:T-1],  (sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])
        @constraint(model, [t=1:T-1], -(sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])

        add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
        add_to_expression!(objective, smoothness_penalty * sum(sent[:,:,1]))
    end
    if setup_cost > 0
        @variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
        @constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[i,j,:])+sum(sent[j,i,:])] in MOI.SOS1([1.0, 1.0]))
        add_to_expression!(objective, setup_cost*sum(setup_dummy))
    end
    @objective(model, Min, objective)

    if send_new_only
        @constraint(model, [t=1:T],
            sum(sent[:,:,t], dims=2) .<= admitted_patients[:,t]
        )
    else
        @constraint(model, [t=1:T],
            sum(sent[:,:,t], dims=2) .<=
                initial_patients - sum(discharged_patients[:,1:min(t,hospitalized_days)], dims=2)
                .+ sum(admitted_patients[:,max(1,t-hospitalized_days):t], dims=2)
                .- sum(sent[:,:,1:t-1], dims=[2,3])
                .+ sum(sent[:,:,max(1,t-hospitalized_days):t-1], dims=[1,3])
        )
    end

    for i = 1:N
        for j = 1:N
            if ~adj_matrix[i,j]
                @constraint(model, sum(sent[i,j,:]) .== 0)
            end
        end
    end

    if sendrecieve_switch_time > 0
        @constraint(model, [i=1:N,t=1:T-1],
            [sum(sent[:,i,t]), sum(sent[i,:,t:min(t+sendrecieve_switch_time,T)])] in MOI.SOS1([1.0, 1.0])
        )
        @constraint(model, [i=1:N,t=1:T-1],
            [sum(sent[:,i,t:min(t+sendrecieve_switch_time,T)]), sum(sent[i,:,t])] in MOI.SOS1([1.0, 1.0])
        )
    end

    flip_sign = (objective == :shortage) ? 1 : -1
    z1, z2 = (objective == :shortage) ? (0, -1) : (-1, 0)
    @constraint(model, [i=1:N,t=1:T],
        obj_dummy[i,t] >= flip_sign * (
            beds[i] - (
                initial_patients[i] - sum(discharged_patients[i,1:min(t,hospitalized_days)])
                + sum(admitted_patients[i,max(1,t-hospitalized_days):t])
                - sum(sent[i,:,1:t+z1])
                + sum(sent[:,i,max(1,t-hospitalized_days):t+z2])
            )
        )
    )

    optimize!(model)
    return model
end;

end;
