module GeographicData

using CSV
using Dates
using DataFrames
using LinearAlgebra: diagm

export adjacencies
export get_counties, get_county_names
export haversine_distance_matrix

basepath = normpath(@__DIR__, "../../")

########################
######### Data #########
########################

function load_states_google(;states::Array{String,1}=String[], method::Symbol=:cities)::Tuple{Array{Int,2},Array{String,1}}
	@assert method in [:states, :cities]
	if (method == :states) fn = "state_dist_google"
	else fn = "state_cities_dist_google"
	end

	dist_df = DataFrame(CSV.File(joinpath(basepath, "data/geography/$(fn).csv")), copycols=true)

	if (method == :cities) state_abbrev_list = map(s -> String(s)[end-1:end], names(dist_df))
	else state_abbrev_list = names(dist_df)[:]
	end

	if !isempty(states)
		@assert states == sort(states)
		ind = [findfirst(x -> x == s, state_abbrev_list) for s in states]
	else
		ind = sortperm(state_abbrev_list)
	end

	state_abbrev_list = state_abbrev_list[ind]
	dist_matrix = Matrix(dist_df)[ind, ind]

	return dist_matrix, state_abbrev_list
end

function load_states_latlong(;states::Array{String,1}=String[])::Tuple{Array{Float64,2},Array{String,1}}
	state_data = DataFrame(CSV.File(joinpath(basepath, "data/geography/states.csv")), copycols=true)
	if !isempty(states) filter!(row -> row.abbrev in states, state_data) end
	sort!(state_data, :abbrev)
	latlong = hcat(state_data.lat[:], state_data.long[:])
	state_abbrev = state_data.abbrev[:]
	return latlong, state_abbrev
end

function load_counties_latlong(;counties::Array{Int,1}=Int[])::Tuple{Array{Float64,2},Array{Int,1}}
	county_data = DataFrame(CSV.File(joinpath(basepath, "data/geography/counties.csv")), copycols=true)
	if !isempty(counties)
		@assert counties == sort(counties)
		filter!(row -> row.fips in counties, county_data)
	end
	sort!(county_data, :fips)
	latlong = hcat(county_data.lat[:], county_data.long[:])
	county_fips = county_data.fips[:]
	return latlong, county_fips
end

########################
###### Adjacency #######
########################

function adjacencies(locations::Array; level::Symbol=:state, source::Symbol=:google, threshold::Real=4, self_edges::Bool=false)::BitArray{2}
	if source == :fullyconnected
		return fully_connected(length(locations), self_edges=self_edges)
	elseif level == :state && source == :google
		dist_matrix, all_states = load_states_google(states=locations)
		@assert all_states == locations
	elseif level == :state && source == :original
		error("Not yet implemented.")
	elseif level == :state && source == :latlong
		latlong, all_states = load_states_latlong(states=locations)
		@assert all_states == locations
		dist_matrix = haversine_distance_matrix(latlong)
	elseif level == :county
		latlong, all_fips = load_counties_latlong(counties=locations)
		@assert all_fips == locations
		dist_matrix = haversine_distance_matrix(latlong)
	elseif level == :hospital
		dist_matrix = haversine_distance_matrix(locations)
	elseif level == :other
		dist_matrix = haversine_distance_matrix(locations)
	else
		error("Invalid parameters to compute_adjacencies.")
	end

	if self_edges
		adj_matrix = 0 .<= dist_matrix .<= 3600 * threshold
	else
		adj_matrix = 0 .<  dist_matrix .<= 3600 * threshold
	end

	return adj_matrix
end

########################
###### Haversine #######
########################

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

function haversine_distance_matrix(locations::Array{Float64,2}; dist_type=:time, speed_kph=100)
	N = size(locations, 1)
	distancematrix = zeros(Float32, N, N)
	for i in 1:N
		for j in i+1:N
			dist = haversine_distance(locations[i,:]..., locations[j,:]..., dist_type=dist_type, speed_kph=speed_kph)
			distancematrix[i,j] = dist
			distancematrix[j,i] = dist
		end
	end
	return distancematrix
end

########################
######## Other #########
########################

function fully_connected(n::Int; self_edges::Bool=false)
	if (self_edges) return BitArray(ones(Bool, n, n)) end
	adj = BitArray(ones(Bool, n, n) - diagm(ones(Bool, n)))
	return adj
end

function get_counties(states::Array{String,1})
	@assert states == sort(states)
	county_data = DataFrame(CSV.File(joinpath(basepath, "data/geography/counties.csv")), copycols=true)
	filter!(row -> row.state_abbrev in states, county_data)
	sort!(county_data, :state_abbrev)
	return county_data.fips[:]
end

function get_county_names(counties::Array{Int,1})
	@assert counties == sort(counties)
	county_data = DataFrame(CSV.File(joinpath(basepath, "data/geography/counties.csv")), copycols=true)
	filter!(row -> row.fips in counties, county_data)
	sort!(county_data, :fips)
	names = map(c -> c.county * " " * c.state_abbrev, eachrow(county_data))
	return names
end

end;
