module mobile_phone_activity_in_Milan

using GeoJSON, DataFrames, DelimitedFiles, Plots

export load_grid, load_telecom_data, fill_missing_data

include("data_loading.jl")

end
