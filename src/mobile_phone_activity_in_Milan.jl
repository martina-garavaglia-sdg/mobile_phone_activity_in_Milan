module mobile_phone_activity_in_Milan

using JSON, DataFrames, DataFramesMeta, DelimitedFiles, Plots, CSV, StatsBase

export load_telecom_data, fill_missing_data, empty_dataframe, make_series, normalize_dataframe, standardize_dataframe, make_film, is_best, split_train_test

include("data_loading.jl")
include("data_processing.jl")
include("metrics.jl")

end
