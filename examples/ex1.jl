using mobile_phone_activity_in_Milan

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

#@show plot(data1.datetime, data1.smsin)

