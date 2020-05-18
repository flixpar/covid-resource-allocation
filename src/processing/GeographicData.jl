module GeographicData

using CSV
using Dates
using DataFrames
using LinearAlgebra: diagm

export compute_adjacencies, fully_connected
export haversine_distance_matrix

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

function fully_connected(n::Int; self_edges::Bool=false)
    if (self_edges) return BitArray(ones(Bool, n, n)) end
    adj = BitArray(ones(Bool, n, n) - diagm(ones(Bool, n)))
    return adj
end

function haversine_distance(lat1, lon1, lat2, lon2; dist_type=:time, speed_kph=100)
    R = 6371e3

    φ1 = lat1 * π/180
    φ2 = lat2 * π/180
    Δφ = (lat2-lat1) * π/180
    Δλ = (lon2-lon1) * π/180

    a = (sin(Δφ/2) * sin(Δφ/2)) + (cos(φ1) * cos(φ2) * sin(Δλ/2) * sin(Δλ/2))
    c = 2 * atan(sqrt(a), sqrt(1-a))

    dist = R * c

    if dist_type == :time
        dist = dist / 1000 / speed_kph * 3600
    end

    return dist
end

function haversine_distance_matrix(locations::Array{Float64,2})
    N = size(locations, 1)
    distancematrix = zeros(Float32, N, N)
    for i in 1:N
        for j in i+1:N
            dist = haversine_time(locations[i,:]..., locations[j,:]...)
            distancematrix[i,j] = dist
            distancematrix[j,i] = dist
        end
    end
    return distancematrix
end

end;
