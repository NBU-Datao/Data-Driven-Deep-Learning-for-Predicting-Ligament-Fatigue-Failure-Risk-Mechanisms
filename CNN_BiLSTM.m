clc;
clear 
close all
addpath('..\') % Add parent directory to path

pwd
X = xlsread('dataset.xlsx');  % Load wind farm prediction data

n_in = 5;  % Number of input time steps
n_out = 1; % Single-step prediction; must be set to 1, otherwise an error will occur
or_dim = size(X,2);       % Record original feature dimension
num_samples = 2000;       % Generate 2000 samples
scroll_window = 2;        % Sliding window step (1: next data from 2nd row, 2: from 3rd row, etc.)
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);


% Train-test split

num_size = 0.8;                                % Ratio of training data
num_train_s = round(num_size * num_samples);   % Number of training samples

P_train = res(1: num_train_s,1);
P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
T_train = res(1: num_train_s,2);
T_train = cell2mat(T_train)';

P_test = res(num_train_s+1: end,1);
P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
T_test = res(num_train_s+1: end,2);
T_test = cell2mat(T_test)';

% Data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

vp_train = reshape(p_train,n_in,or_dim,num_train_s);
vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);

vt_train = t_train;
vt_test = t_test;

% Reshape data into 3D format for training

for i = 1:size(P_train,2)
    trainD{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
end

for i = 1:size(p_test,2)
    testD{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
end

targetD =  t_train;
targetD_test  =  t_test;

numFeatures = size(p_train,1);

% Define CNN-BiLSTM network architecture
layers0 = [ ...
    sequenceInputLayer([numFeatures,1,1],'name','input')   % Input layer
    sequenceFoldingLayer('name','fold')                    % Sequence folding layer (prepares for 2D conv)
    
    % CNN feature extraction
    convolution2dLayer([3,1],16,'Stride',[1,1],'name','conv1')  % 2D convolution layer
    batchNormalizationLayer('name','batchnorm1')                % Batch normalization layer
    reluLayer('name','relu1')                                   % ReLU activation layer
    
    % Pooling layer
    maxPooling2dLayer([2,1],'Stride',2,'Padding','same','name','maxpool')  % Max pooling layer

    % Unfold sequence back to time-series
    sequenceUnfoldingLayer('name','unfold')                     % Restore sequence
    
    % Flatten
    flattenLayer('name','flatten')
    
    bilstmLayer(25,'Outputmode','last','name','hidden1')        % BiLSTM layer
    dropoutLayer(0.1,'name','dropout_1')                        % Dropout layer with 0.1 rate

    fullyConnectedLayer(1,'name','fullconnect')                 % Fully connected layer
    regressionLayer('Name','output')                            % Regression output layer
];

lgraph0 = layerGraph(layers0);
lgraph0 = connectLayers(lgraph0,'fold/miniBatchSize','unfold/miniBatchSize');


% Set training hyperparameters
options0 = trainingOptions('adam', ...                % Optimizer: Adam
    'MaxEpochs', 150, ...                             % Max training epochs
    'GradientThreshold', 1, ...                       % Gradient clipping threshold
    'InitialLearnRate', 0.01, ...                     % Initial learning rate
    'LearnRateSchedule', 'piecewise', ...             % Learning rate schedule
    'LearnRateDropPeriod',100, ...                    % Drop after 100 epochs
    'LearnRateDropFactor',0.01, ...                   % Drop factor
    'L2Regularization', 0.001, ...                    % L2 regularization
    'ExecutionEnvironment', 'cpu',...                 % Training on CPU
    'Verbose', 1, ...                                 % Display training info
    'Plots', 'none');                                 % No training plots

% Start training
tic
net = trainNetwork(trainD,targetD',lgraph0,options0);
toc

% Predict
t_sim1 = predict(net, trainD); 
t_sim2 = predict(net, testD); 

% Reverse normalization
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

% Convert data format
T_sim1 = double(T_sim1);
T_sim2 = double(T_sim2);

CNNBILSTM_TSIM1 = T_sim1';
CNNBILSTM_TSIM2 = T_sim2';
save CNNBILSTM CNNBILSTM_TSIM1 CNNBILSTM_TSIM2


% Evaluation metrics
disp('…… Training set error metrics ……')
[mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_train1);
hold on
plot(T_sim1')
legend('Ground Truth','Prediction')
title('CNN-BiLSTM Training Prediction Performance')
xlabel('sample point')
ylabel('Results')

disp('…… Testing set error metrics ……')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_test2);
hold on
plot(T_sim2')
legend('Ground Truth','Prediction')
title('CNN-BiLSTM Testing Prediction Performance')
xlabel('sample point')
ylabel('Results')

figure('Position',[200,300,600,200])
plot(T_sim2'-T_test2)
title('CNN-BiLSTM Prediction Error Curve')
xlabel('sample point')
ylabel('Results')
