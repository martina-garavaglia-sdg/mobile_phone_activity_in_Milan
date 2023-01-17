module mobile_phone_activity_in_Milan

using GeoJSON, DataFrames, DelimitedFiles, Plots, CSV, DataFramesMeta

export load_grid, load_telecom_data, fill_missing_data, empty_dataframe, make_series

include("data_loading.jl")
include("data_processing.jl")

end
