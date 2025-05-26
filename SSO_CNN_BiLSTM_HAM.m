clc;
clear 
close all
addpath('..\') % Load parent directory
addpath('SSA_package\'); % Add Sparrow Search Algorithm package

% Load dataset
X = xlsread('Dataset.xlsx');

% Parameters setup
n_in = 5;  % Input time steps
n_out = 1; % Output steps (must be 1)
or_dim = size(X,2); % Feature dimensions
num_samples = 2000; % Number of samples
scroll_window = 2; % Sliding window step

% Data preparation
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);

% Train-test split
num_size = 0.8; % Training set ratio
num_train_s = round(num_size * num_samples); % Training samples count

P_train = res(1:num_train_s,1);
P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
T_train = res(1:num_train_s,2);
T_train = cell2mat(T_train)';

P_test = res(num_train_s+1:end,1);
P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
T_test = res(num_train_s+1:end,2);
T_test = cell2mat(T_test)';

% Data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

% Reshape data for training
for i = 1:size(P_train,2)
    trainD{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
end

for i = 1:size(p_test,2)
    testD{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
end

targetD = t_train;
targetD_test = t_test;
numFeatures = size(p_train,1);

%% SSA Optimization for Hyperparameters
% Define optimization problem
problem.cost_func = @(x) CNNBiLSTM_Attention_Fitness(x, trainD, targetD, numFeatures);
problem.n_var = 5; % Number of variables to optimize
problem.var_min = [0.001, 0.0001, 10, 0.1, 10]; % Lower bounds [lr, L2, LSTM units, dropout, attention channels]
problem.var_max = [0.1, 0.01, 100, 0.5, 20];    % Upper bounds

% SSA parameters
params.max_iter = 10;      % Maximum iterations
params.n_pop = 20;        % Population size
params.pd = 0.2;          % Discoverers ratio
params.sd = 0.1;          % Scouts ratio
params.progress_bar = true;
params.plot_progress = true;

% Run SSA optimization
[best_params, best_cost, convergence_curve] = SSA(problem, params);

% Extract best parameters
best_learningRate = best_params(1);
best_L2Regularization = best_params(2);
best_LSTMUnits = round(best_params(3));
best_dropoutRate = best_params(4);
best_attentionChannels = round(best_params(5));

disp('=== SSA Optimization Results ===');
disp(['Best Learning Rate: ', num2str(best_learningRate)]);
disp(['Best L2 Regularization: ', num2str(best_L2Regularization)]);
disp(['Best LSTM Units: ', num2str(best_LSTMUnits)]);
disp(['Best Dropout Rate: ', num2str(best_dropoutRate)]);
disp(['Best Attention Channels: ', num2str(best_attentionChannels)]);
disp(['Best RMSE: ', num2str(best_cost)]);

%% Build and train model with optimized parameters
layers0 = [ ...
    sequenceInputLayer([numFeatures,1,1],'name','input')
    sequenceFoldingLayer('name','fold')
    
    % CNN feature extraction
    convolution2dLayer([3,1],16,'Stride',[1,1],'name','conv1')
    batchNormalizationLayer('name','batchnorm1')
    reluLayer('name','relu1')
    maxPooling2dLayer([2,1],'Stride',2,'Padding','same','name','maxpool')
    sequenceUnfoldingLayer('name','unfold')
    flattenLayer('name','flatten')
    
    bilstmLayer(best_LSTMUnits,'Outputmode','last','name','hidden1')
    selfAttentionLayer(1,best_attentionChannels) % Self-attention layer with optimized channels
    dropoutLayer(best_dropoutRate,'name','dropout_1')
    fullyConnectedLayer(1,'name','fullconnect')
    regressionLayer('Name','output')];

lgraph0 = layerGraph(layers0);
lgraph0 = connectLayers(lgraph0,'fold/miniBatchSize','unfold/miniBatchSize');

% Training options with optimized parameters
options0 = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', best_learningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 100, ...
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization', best_L2Regularization, ...
    'ExecutionEnvironment', 'cpu',...
    'Verbose', 1, ...
    'Plots', 'none');

% Train model
disp('=== Training SSA-CNN-BiLSTM-Attention Model ===');
tic
net = trainNetwork(trainD, targetD', lgraph0, options0);
toc

% Prediction
t_sim1 = predict(net, trainD); 
t_sim2 = predict(net, testD); 

% Denormalize
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

% Convert data format
T_sim1 = double(T_sim1);
T_sim2 = double(T_sim2);

% Save results
SSA_CNNBILSTM_ATTENTION_TSIM1 = T_sim1';
SSA_CNNBILSTM_ATTENTION_TSIM2 = T_sim2';
save SSA_CNNBILSTM_ATTENTION SSA_CNNBILSTM_ATTENTION_TSIM1 SSA_CNNBILSTM_ATTENTION_TSIM2

% Evaluation metrics
disp('...... Training Set Metrics ......')
[mae1,rmse1,mape1,error1] = calc_error(T_train1,T_sim1');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_train1); hold on; plot(T_sim1')
legend('Actual','Predicted')
title('SSA-CNN-BiLSTM-Attention Training Performance')
xlabel('Sample Point'); ylabel('Value')

disp('...... Test Set Metrics ......')
[mae2,rmse2,mape2,error2] = calc_error(T_test2,T_sim2');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_test2); hold on; plot(T_sim2')
legend('Actual','Predicted')
title('SSA-CNN-BiLSTM-Attention Test Performance')
xlabel('Sample Point'); ylabel('Value')

figure('Position',[200,300,600,200])
plot(T_sim2'-T_test2)
title('SSA-CNN-BiLSTM-Attention Prediction Errors')
xlabel('Sample Point'); ylabel('Error')