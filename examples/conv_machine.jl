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

x, y = split_x_y(data1, data2, data3, data4, data5, data6, data7)

x = dropdims(x, dims=3)
y = dropdims(y, dims=3)

x = Flux.unsqueeze(x, dims=4)
y = Flux.unsqueeze(y, dims=4)
x = Flux.unsqueeze(x, dims=5)
y = Flux.unsqueeze(y, dims=5)


# Loading
data = DataLoader((x, y));


# (100, 100, 144, 1, 1)
# Dimensions
dimensions = [1,2,4,8];

machine = ConvMachine(dimensions, sigmoid; pad=(1,1,1,1,10,0)); #tanh, 5

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

for epoch in 171:300

    # Train
    Flux.train!(loss, params, data, opt)

    # Saving loss for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x, y))
    @show loss(x, y)

    # Saving the best parameters
    # if epoch > 1
    #     if is_best(loss_on_train[epoch-1], loss_on_train[epoch])
    #         best_params = params
    #     end
    # end
end

# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);


############################################
############# Visualization ################
############################################

# Loss
plot(epochs, loss_on_train, c=:blue, lw=2, ylims = (0,2));
title!("Convolutional machine");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("visualization/losses/loss_conv_machine_150.png");


m = model(x)
# Heatmap for the last hour 
heatmap(m[3:98,3:98,144], color=:thermal, clims=(-0.01, 0.1)) # removing margin
#heatmap(m[:,:,132], color=:thermal)
savefig("visualization/heatmaps/heatmap_it170.png");


heatmap(m[3:98,3:98,132], color=:thermal, clims=(-0.01, 0.1))
savefig("visualization/gif/heatmap24.png");




##############################################
################# Metrics ####################
##############################################

h = 24
d_test = 5

# m = model(x)
pred = model(x)[:,:,h*(d_test-1):h*d_test,:,:]; # prediction day 7
gt = y[:,:,h*(d_test-1):h*d_test,:,:]; # day 7
pred_naive1 = zeros(size(gt));
pred_naive2 = y[:,:,h*(d_test-2):h*(d_test-1),:]; # day 6

##### Model naive 1: comparing my data to zero model usong mse and mae
error_naive1_mse_test = Flux.Losses.mse(pred_naive1, gt)
error_naive1_mae_test = Flux.Losses.mae(pred_naive1, gt)

# Model naive 2: comparing my data to my data the day before
error_naive2_mse_test = Flux.Losses.mse(pred_naive2, gt)
error_naive2_mae_test = Flux.Losses.mae(pred_naive2, gt)

# Test error model
error_test_set_mse = Flux.Losses.mse(pred, gt)
error_test_set_mae = Flux.Losses.mae(pred, gt)



################################################################
######### Average on space, error for every hour ###############
################################################################


# Model Naive 1
error_naive1_hour_test = []
for hour in 1:h
    push!(error_naive1_hour_test, Flux.Losses.mae(zeros(100,100,1,1,1), y[:,:,h*(d_test-1)+hour,:,:])) 
end
error_naive1_hour_test

# Model Naive 2
error_naive2_hour_test = []
for hour in 1:h
    push!(error_naive2_hour_test, Flux.Losses.mae(y[:,:,h*(d_test-1)+hour,:], y[:,:,h*(d_test-1)+hour-1,:,:])) 
end
error_naive2_hour_test

#####

plot(error_naive2_hour_test)
plot!(error_hour_mae_test)
plot!(error_naive1_hour_test)

# Test error

error_hour_mae_test = []
for hour in 1:h
    push!(error_hour_mae_test, Flux.Losses.mae(m[:,:,h*(d_test-1)+hour,:,:], y[:,:,h*(d_test-1)+hour,:,:])) 
end
error_hour_mae_test


###################################################################
############# Average on time, error for every cell ###############
###################################################################

# Model naive 1

error_naive1_cell_test = []

for i in 1:100
    for j in 1:100
        push!(error_naive1_cell_test, Flux.Losses.mae(zeros(size(y[i,j,(d_test-1)*h:d_test*h,:,:])), y[i,j,(d_test-1)*h:d_test*h,:,:]))
    end
end
error_naive1_cell_test

# Model naive 2

error_naive2_cell_test = []

for i in 1:100
    for j in 1:100
        push!(error_naive2_cell_test, Flux.Losses.mae(y[i,j,h*(d_test-2):h*(d_test-1),:], y[i,j,h*(d_test-1):h*d_test,:,:]))
    end
end
error_naive2_cell_test

# Test error model

error_cell_mae_test = []

for i in 1:100
    for j in 1:100
        push!(error_cell_mae_test, Flux.Losses.mae(m[i,j,h*(d_test-1):h*d_test,:], y[i,j,h*(d_test-1):h*d_test,:,:]))
    end
end
error_cell_mae_test
