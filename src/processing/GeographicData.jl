module GeographicData

using CSV
using Dates
using DataFrames

export compute_adjacencies

basepath = normpath(@__DIR__, "../../")

########################
###### Adjacency #######
########################

let
    global dist_matrix, state_abbrev_list

    dist_df = CSV.read(joinpath(basepath, "data/geography/state_cities_dist_google.csv"), copycols=true)
    dist_data = Matrix(dist_df)

    state_abbrev_list = map(s -> String(s)[end-1:end], names(dist_df))
    order = sortperm(state_abbrev_list)
    state_abbrev_list = state_abbrev_list[order]

    dist_matrix = dist_data[order, order]
end

function compute_adjacencies(states; hrs::Real=4, self_edges::Bool=false)
    @assert states == sort(states)
    ind = [findfirst(x -> x == s, state_abbrev_list) for s in states]
    dist = dist_matrix[ind, ind]

    dist_threshold = 3600 * hrs
    adj_matrix = 0 .< dist .<= dist_threshold

    if self_edges
        adj = BitArray(adj + diagm(ones(Bool, length(states))))
    end

    return adj_matrix
end

end;
