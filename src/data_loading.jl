using JSON, DataFrames, DelimitedFiles, CSV

# Loading telecommunication data

function load_telecom_data(df_format = true)
    if df_format 
        data1 = DataFrame(CSV.File("data/sms-call-internet-mi-2013-11-01.csv"));
        data2 = DataFrame(CSV.File("data/sms-call-internet-mi-2013-11-02.csv"));
        data3 = DataFrame(CSV.File("data/sms-call-internet-mi-2013-11-03.csv"));
        data4 = DataFrame(CSV.File("data/sms-call-internet-mi-2013-11-04.csv"));
        data5 = DataFrame(CSV.File("data/sms-call-internet-mi-2013-11-05.csv"));
        data6 = DataFrame(CSV.File("data/sms-call-internet-mi-2013-11-06.csv"));
        data7 = DataFrame(CSV.File("data/sms-call-internet-mi-2013-11-07.csv"));
    else
        data1 = readdlm("data/sms-call-internet-mi-2013-11-01.csv",  ',');
        data2 = readdlm("data/sms-call-internet-mi-2013-11-02.csv",  ',');
        data3 = readdlm("data/sms-call-internet-mi-2013-11-03.csv",  ',');
        data4 = readdlm("data/sms-call-internet-mi-2013-11-04.csv",  ',');
        data5 = readdlm("data/sms-call-internet-mi-2013-11-05.csv",  ',');
        data6 = readdlm("data/sms-call-internet-mi-2013-11-06.csv",  ',');
        data7 = readdlm("data/sms-call-internet-mi-2013-11-07.csv",  ',');
    end
    return data1, data2, data3, data4, data5, data6, data7
end

# Fill missing data with zero value

function fill_missing_data(data::Any)
    if typeof(data) == Matrix{Any}
        for i in 2:size(data)[1]
            for j in 1:size(data)[2]
                if data[i,j] == ""
                    data[i,j] = Float64(0)
                end
            end
        end
    else
        [replace!(data[!,i], missing => 0) for i in names(data)]
    end
    return data
end
