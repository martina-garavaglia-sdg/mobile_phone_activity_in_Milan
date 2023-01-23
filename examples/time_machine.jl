using mobile_phone_activity_in_Milan
using DataFramesMeta, JSON
using Flux
using Flux: @epochs, train!
using Flux.Data: DataLoader
using Plots
using ParametricMachinesDemos

# Data loading and processing
grid = JSON.parsefile("data/milano-grid.geojson");

data1, data2, data3, data4, data5, data6, data7 = load_telecom_data(true);

x_train, x_test, y_train, y_test = split_train_test(data1, data2, data3, data4, data5, data6, data7);

x_train = dropdims(x_train, dims=3);
y_train = dropdims(y_train, dims=3);
x_test = dropdims(x_test, dims=3);
y_test = dropdims(y_test, dims=3);

# Loading
train_data = DataLoader((x_train, y_train); batchsize = 16, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 16, shuffle = true);


# Dimensions
dimensions = [16, 64, 32];

machine = RecurMachine(dimensions, sigmoid; pad=1, timeblock=10);

model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 100));

model = cpu(model)

opt = ADAM(0.01);

params = Flux.params(model);

# Loss function
loss(x,y) = Flux.Losses.mse(model(x), y)

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
best_params = Float32[]

for epoch in 1:10

    # Train
    Flux.train!(loss, params, train_data, opt)

    # Show the sum of the gradients
    gs = gradient(params) do
        loss(x_train, y_train)
    end
    #@show [norm(g,2) for g in gs] # gradient's norm
    
    # Saving losses and accuracies for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(loss_on_test, loss(x_test, y_test))
    @show loss(x_train, y_train)
    @show loss(x_test, y_test)

    # Saving the best parameters
    if epoch > 1
        if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
            best_params = params
        end
    end
end

# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);


# Visualization
plot(epochs, loss_on_train, lab="Training", c=:black, lw=2, ylims = (0,0.2));
plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2, ylims = (0,0.2));
title!("Time machine");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("loss_time_machine.png");









### PROVE
# data1 = fill_missing_data(data1);
# data2 = fill_missing_data(data2);
# data3 = fill_missing_data(data3);
# data4 = fill_missing_data(data4);
# data5 = fill_missing_data(data5);
# data6 = fill_missing_data(data6);
# data7 = fill_missing_data(data7);


# Normalize/standardize
# ds1 = standardize_dataframe(data7);
# m1 = fill_missing_data(ds1);
# # Columns type conversion

# # data1[!,:smsin] = convert.(Float64,data1[!,:smsin])
# # data1[!,:smsout] = convert.(Float64,data1[!,:smsout])
# # data1[!,:callin] = convert.(Float64,data1[!,:callin])
# # data1[!,:callout] = convert.(Float64,data1[!,:callout])
# # data1[!,:internet] = convert.(Float64,data1[!,:internet])

# # Grouped by and combine
# grouped_data1 = groupby(m1, [:datetime, :CellID]);
# d1 = @combine(grouped_data1, :max_i = maximum(:smsin)) #, :smsout, :callin, :callout, :internet]));  # ! non dovrei prendere solo smsin, ma tutte le variabili
#                                                         # con le parentesi quadre non fa la stessa cosa! capire il perchè
# # Now I want to make a "film" creating matrix with: same time, dispose by cellID, with intensity (value) the values in max_smsin

# g1 = groupby(d1, [:datetime]); # here

# d = make_film(g1);

# NB: la griglia va dall'alto verso il basso mentre quella "vera" va al contrario


