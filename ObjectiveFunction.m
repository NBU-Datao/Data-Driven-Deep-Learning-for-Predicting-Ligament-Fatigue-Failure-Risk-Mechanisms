
function [MAPE,T_sim]= objectiveFunction(x,f_,vp_train,vt_train,vp_test,T_test,ps_output)

%% Convert input optimization parameters to hyperparameters
learning_rate = x(1);            % Learning rate
NumNeurons = round(x(2));        % Number of BiLSTM neurons
keys = round(x(3));              % Number of keys in self-attention mechanism
L2Regularization = x(4);         % L2 regularization parameter
setdemorandstream(pi);

layers0 = [ ...
    % Input features
    sequenceInputLayer([f_,1,1],'name','input')   % Input layer
    sequenceFoldingLayer('name','fold')           % Sequence folding layer for time-step convolution

    % CNN feature extraction
    convolution2dLayer([3,1],16,'Stride',[1,1],'name','conv1')  % Convolution layer: filter size 3x1, 16 filters
    batchNormalizationLayer('name','batchnorm1')                % Batch normalization to speed up training
    reluLayer('name','relu1')                                   % ReLU activation to introduce non-linearity

    % Pooling layer
    maxPooling2dLayer([2,1],'Stride',2,'Padding','same','name','maxpool')  % Max pooling layer with 2x1 kernel and stride 2

    % Unfold layer
    sequenceUnfoldingLayer('name','unfold')       % Unfold sequence after convolution

    % Flatten
    flattenLayer('name','flatten')

    bilstmLayer(NumNeurons,'Outputmode','last','name','hidden1') 
    selfAttentionLayer(1,keys)                    % Self-attention layer with 1 head and "keys" key/query channels
    dropoutLayer(0.1,'name','dropout_1')          % Dropout layer with drop rate 0.1

    fullyConnectedLayer(1,'name','fullconnect')   % Fully connected layer (affects output dimension)
    regressionLayer('Name','output')              % Regression output layer
];

lgraph0 = layerGraph(layers0);
lgraph0 = connectLayers(lgraph0,'fold/miniBatchSize','unfold/miniBatchSize');

%% Set the hyperparameters for training
options = trainingOptions('adam', ...                 % Optimization algorithm: Adam
    'MaxEpochs', 30, ...                              % Maximum number of training epochs
    'GradientThreshold', 1, ...                       % Gradient clipping threshold
    'InitialLearnRate', learning_rate, ...            % Initial learning rate
    'L2Regularization', L2Regularization, ...         % L2 regularization parameter
    'ExecutionEnvironment', 'cpu',...                 % Execution environment
    'Verbose', 1, ...                                 % Verbosity flag
    'Plots', 'none');                                 % Disable training plots

% Train the model
net = trainNetwork(vp_train, vt_train, lgraph0, options);

% analyzeNetwork(net); % View model structure (optional)

%% Testing and evaluation
t_sim = net.predict(vp_test);  

% Reverse normalization
T_sim = mapminmax('reverse', t_sim, ps_output);

% Convert data format
T_sim = double(T_sim);
T_sim = T_sim';

%% Calculate error
MAPE = sum(abs((T_sim - T_test)./T_test)) / length(T_test);  % Mean Absolute Percentage Error
display(['MAPE of current batch: ', num2str(MAPE)]);
end