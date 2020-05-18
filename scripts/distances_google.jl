using HTTP
using JSON

function google_distance_matrix_small(locations::Array{String,1})
    if !haskey(ENV, "GAPI_KEY")
        error("Google API key not found.")
        return
    end

    locs = [foldl(replace, [", "=>"+", " "=>"+", ","=>"+"], init=s) for s in locations]
    locs_string = join(locs, "|")

    key = ENV["GAPI_KEY"]
    url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=$(locs_string)&destinations=$(locs_string)&key=$(key)"

    response = HTTP.request("GET", url)
    results = JSON.parse(String(response.body))

    N = length(locations)
    distancematrix = zeros(Int, N, N)
    for i in 1:N
        for j in 1:N
            if (i == j) continue end
            if (results["rows"][i]["elements"][j]["status"] != "OK") distancematrix[i,j] = typemax(Int); continue; end
            distancematrix[i,j] = results["rows"][i]["elements"][j]["duration"]["value"]
        end
    end

    return distancematrix
end

function google_distance_matrix_large(locations::Array{String,1})
    if !haskey(ENV, "GAPI_KEY")
        error("Google API key not found.")
        return
    end
    key = ENV["GAPI_KEY"]

    locs = [foldl(replace, [", "=>"+", " "=>"+", ","=>"+"], init=s) for s in locations]

    N = length(locations)
    distancematrix = zeros(Int, N, N)
    for i in 1:N
        for j in i+1:N

            s = locs[i]
            d = locs[j]
            locs_string = s * "|" * d
            url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=$(locs_string)&destinations=$(locs_string)&key=$(key)"

            response = HTTP.request("GET", url)
            results = JSON.parse(String(response.body))

            if (results["rows"][1]["elements"][2]["status"] != "OK") || (results["rows"][2]["elements"][1]["status"] != "OK")
                distancematrix[i,j] = typemax(Int)
                distancematrix[j,i] = typemax(Int)
                continue
            end

            distancematrix[i,j] = results["rows"][1]["elements"][2]["duration"]["value"]
            distancematrix[j,i] = results["rows"][2]["elements"][1]["duration"]["value"]
        end
    end

    return distancematrix
end

function google_distance_matrix_symmetric(locations::Array{String,1})
    if !haskey(ENV, "GAPI_KEY")
        error("Google API key not found.")
        return
    end
    key = ENV["GAPI_KEY"]

    locs = [foldl(replace, [", "=>"+", " "=>"+", ","=>"+"], init=s) for s in locations]

    N = length(locations)
    distancematrix = zeros(Int, N, N)
    for i in 1:N
        for j in i+1:N
            s = locs[i]
            d = locs[j]
            url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=$(s)&destinations=$(d)&key=$(key)"

            response = HTTP.request("GET", url)
            results = JSON.parse(String(response.body))["rows"][1]["elements"][1]

            if results["status"] != "OK"
                distancematrix[i,j] = typemax(Int)
                distancematrix[j,i] = typemax(Int)
                continue
            end

            distancematrix[i,j] = results["duration"]["value"]
            distancematrix[j,i] = results["duration"]["value"]
        end
    end

    return distancematrix
end

function google_distance_matrix(locations::Array{String,1})
    if length(locations) < 10
        return build_distance_matrix_small(locations)
    else
        return google_distance_matrix_symmetric(locations)
    end
end
