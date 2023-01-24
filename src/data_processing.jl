using DataFrames, StatsBase, Flux

# Trasform my data in order to have a temporal series for every CellID

function make_series(df)
    if typeof(df) == DataFrames.DataFrame
        new_df = empty_dataframe()
        for i in 1:10000
            append!(new_df, subset(df, :CellID => ByRow(CellID -> CellID == i)))
        end
    end
    return new_df
end


function empty_dataframe()
    return DataFrame(
        datetime = String[], 
        CellID = Int64[], 
        countrycode = Int64[], 
        smsin = Float64[], 
        smsout = Float64[], 
        callin = Float64[], 
        callout = Float64[], 
        internet = Float64[])
end

# Normalize data
# SKIPMISSING FUNCTION!!
function normalize_dataframe(df::DataFrame)
    d = copy(df)
    mis=0
    for col in 4:8 
        v = []
        for row in 1:nrow(d)
            if typeof(d[row,col]) != Missing
                append!(v, d[row,col])
            end
        end
        m = mean(v)
        s = std(v)
        for row in 1:nrow(d)
            d[row,col] = (d[row,col] - m ) / s  
        end
    end
    return d
end


function standardize_dataframe(df::DataFrame)
    d = copy(df)
    for col in 4:8
        M = maximum(skipmissing(d[:,col]))
        m = minimum(skipmissing(d[:,col]))
        for row in 1:nrow(d)
            d[row, col] = (d[row,col] - m) / (M - m)
        end
    end
    return d
end


function make_film(g_data::Any)

    v = []

    

    for g in 1:length(g_data)
        push!(v, transpose(reshape(g_data[g][:,3], (100,100))))
    end
    v = cat(v..., dims=4)

    return v
end



function make_data(data::DataFrame)
    # standardization
    ds = standardize_dataframe(data);

    # Fill missing data with zero value
    m = fill_missing_data(ds);

    grouped_data = groupby(m, [:datetime, :CellID]);
    d = @combine(grouped_data, :max_ = maximum(:smsin));
    g = groupby(d, [:datetime]); 

    f = make_film(g);

    return f
end


function split_train_test(data1::DataFrame, data2::DataFrame, data3::DataFrame, data4::DataFrame, data5::DataFrame, data6::DataFrame, data7::DataFrame)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    d1 = make_data(data1)
    d2 = make_data(data2)
    d3 = make_data(data3)
    d4 = make_data(data4)
    d5 = make_data(data5)
    d6 = make_data(data6)
    d7 = make_data(data7)

    push!(x_train, d1)
    push!(x_train, d2)
    push!(x_train, d3)
    push!(x_train, d4)
    push!(x_train, d5)
    push!(x_train, d6[:,:,:,1:23])
    x_train = cat(x_train..., dims=4)

    push!(y_train, d1[:,:,:,2:end])
    push!(y_train, d2)
    push!(y_train, d3)
    push!(y_train, d4)
    push!(y_train, d5)
    push!(y_train, d6)
    y_train = cat(y_train..., dims=4)

    x_test = d7[:,:,:,1:23]
    y_test = d7[:,:,:,2:24]

    return x_train, x_test, y_train, y_test
end


function split_x_y(data1::DataFrame, data2::DataFrame, data3::DataFrame, data4::DataFrame, data5::DataFrame, data6::DataFrame, data7::DataFrame)
    x = []
    y = []

    d1 = make_data(data1)
    d2 = make_data(data2)
    d3 = make_data(data3)
    d4 = make_data(data4)
    d5 = make_data(data5)
    d6 = make_data(data6)
    d7 = make_data(data7)

    push!(x, d1)
    push!(x, d2)
    push!(x, d3)
    push!(x, d4)
    push!(x, d5)
    push!(x, d6)
    x = cat(x..., dims=4)

    push!(y, d2)
    push!(y, d3)
    push!(y, d4)
    push!(y, d5)
    push!(y, d6)
    push!(y, d7)
    y = cat(y..., dims=4)

    return x,y

end