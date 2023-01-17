using mobile_phone_activity_in_Milan
using DataFramesMeta

# Data loading and processing
grid = load_grid("data/milano-grid.geojson", true);

data1, data2, data3, data4, data5, data6, data7 = load_telecom_data(true);

data1 = fill_missing_data(data1);
data2 = fill_missing_data(data2);
data3 = fill_missing_data(data3);
data4 = fill_missing_data(data4);
data5 = fill_missing_data(data5);
data6 = fill_missing_data(data6);
data7 = fill_missing_data(data7);

grouped_data = groupby(data1, [:datetime, :CellID])
d = @combine(grouped_data, :max_smsin = maximum(:smsin))


data1[!,:smsin] = convert.(Float64,data1[!,:smsin])
data1[!,:smsout] = convert.(Float64,data1[!,:smsout])
data1[!,:callin] = convert.(Float64,data1[!,:callin])
data1[!,:callout] = convert.(Float64,data1[!,:callout])
data1[!,:internet] = convert.(Float64,data1[!,:internet])

