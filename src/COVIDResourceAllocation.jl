module COVIDResourceAllocation

include("models/PatientAllocation.jl")
include("models/ReusableResourceAllocation.jl")
include("models/DisposableResourceAllocation.jl")

include("processing/BedsData.jl")
include("processing/ForecastData.jl")
include("processing/GeographicData.jl")
include("processing/NurseData.jl")

include("util/PatientAllocationResults.jl")
include("util/NurseAllocationResults.jl")

import .PatientAllocation: patient_allocation
import .ReusableResourceAllocation: reusable_resource_allocation
import .ForecastData: forecast
import .BedsData: n_beds
import .GeographicData: adjacencies
import .NurseData: n_nurses
import .PatientAllocationResults
import .NurseAllocationResults

export patient_allocation, reusable_resource_allocation
export forecast, adjacencies, n_beds, n_nurses
export PatientAllocationResults, NurseAllocationResults

end
