module COVIDResourceAllocation

include("models/PatientAllocation.jl")
include("models/ReusableResourceAllocation.jl")
include("models/DisposableResourceAllocation.jl")

include("procecssing/BedsData.jl")
include("procecssing/ForecastData.jl")
include("procecssing/GeographicData.jl")
include("procecssing/NurseData.jl")

export patient_allocation, reusable_resource_allocation
export forecast, adjacencies, n_beds

end
