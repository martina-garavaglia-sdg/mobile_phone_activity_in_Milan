using DataFrames, StatsBase

# Trasform my data in order to have a temporal series for every CellID

function make_series(df)
    # The input must be a dataframe (TODO: case with matrix)
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

function normalize_dataframe(df)
    #foreach(c -> c .= (c .- mean(c)) ./ std(c), eachcol(df))
    for col in 4:8 
        m = mean(d[:,col])
        s = std(d[:,col])
        for row in 1:nrow(d)
            d[row,col] = (d[row,col] - m ) / s
        end
    end
    return d
end
# ??? non funziona


function norm_data(df)
    

end