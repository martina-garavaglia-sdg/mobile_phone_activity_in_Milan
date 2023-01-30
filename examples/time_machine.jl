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

# x e y: x i primi 6 giorni, y il settimo
x, y = split_x_y(data1, data2, data3, data4, data5, data6, data7)

x = dropdims(x, dims=3)
y = dropdims(y, dims=3)

x = Flux.unsqueeze(x, dims=4)
y = Flux.unsqueeze(y, dims=4)
x = Flux.unsqueeze(x, dims=5)
y = Flux.unsqueeze(y, dims=5)

# x_train, x_test, y_train, y_test = split_train_test(data1, data2, data3, data4, data5, data6, data7);

# x_train = dropdims(x_train, dims=3);
# y_train = dropdims(y_train, dims=3);
# x_test = dropdims(x_test, dims=3);
# y_test = dropdims(y_test, dims=3);

# Loading
data = DataLoader((x, y));
#test_data = DataLoader((x_test, y_test); batchsize = 16, shuffle = true);

# (100, 100, 144, 1, 1)
# Dimensions
dimensions = [1,2,4,8];

machine = ConvMachine(dimensions, sigmoid; pad=(1,1,1,1,10,0));

model = Flux.Chain(machine, Conv((1,1,1), sum(dimensions) => 1));

model = cpu(model);

opt = ADAM(0.01);

params = Flux.params(model);

# Loss function
loss(x,y) = Flux.Losses.mse(model(x), y); #mse

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
best_params = Float32[]

for epoch in 1:100

    # Train
    Flux.train!(loss, params, data, opt)

    # Saving loss for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x, y))
    @show loss(x, y)

    # Saving the best parameters
    if epoch > 1
        if is_best(loss_on_train[epoch-1], loss_on_train[epoch])
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
plot(epochs, loss_on_train, c=:black, lw=2, ylims = (0,0.2));
title!("Time machine");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("loss_time_machine.png");


heatmap(model(x)[3:98,3:98,144], color=:thermal) #togliendo il bordo
#heatmap(model(x)[:,:,144], color=:thermal)
savefig("heatmap_time_machine3.png");


for i in 1:30
    Flux.train!(loss, params, data, opt)
    @show loss(x,y)
end



# Metriche

# Modello Naive 1 : compare 0 prediction model to my data
#TRAIN
error_naive1_train = Flux.Losses.mse(x, zeros(100,100,144,1,1))
#TEST
error_naive1_test = Flux.Losses.mse(y, zeros(100,100,144,1,1))

# Modello Naive 2 : compare my data with my train data the day before
#TRAIN
error_naive2_train = Flux.Losses.mse(x[:,:,1:120,:,:], x[:,:,25:144,:,:])
#TEST
error_naive2_test = Flux.Losses.mse(y[:,:,1:120,:,:], y[:,:,25:144,:,:])


# Train set error: 
error_train_set = Flux.Losses.mse(x[:,:,25:144,:,:], model(x)[:,:,1:120,:,:])

# Test set error:
error_test_set = Flux.Losses.mse(y[:,:,120:144,:,:], model(x)[:,:,120:144,:,:])

