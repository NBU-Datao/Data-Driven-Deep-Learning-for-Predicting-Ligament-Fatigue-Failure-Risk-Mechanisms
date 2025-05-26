clc;
clear 
close all
addpath(genpath(pwd))
X = xlsread('dataset.xlsx');
X = X(5665:8640,:);  % Select March data
num_samples = length(X);                            % Number of samples
kim = 10;                      % Time delay (use kim historical steps as input)
zim =  1;                      % Predict after zim time steps
or_dim = size(X,2);

% Reconstruct dataset
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,end)];
end

% Train-test split
outdim = 1;                                  % Last column is output
num_size = 0.9;                              % Training set ratio
num_train_s = round(num_size * num_samples); % Number of training samples
f_ = size(res, 2) - outdim;                  % Input feature dimension

P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

% Data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

for i = 1:size(P_train,2)
    trainD{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
end

for i = 1:size(p_test,2)
    testD{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
end

targetD =  t_train;
targetD_test  =  t_test;

numFeatures = size(p_train,1);

layers0 = [ ...
    sequenceInputLayer([numFeatures,1,1],'name','input')   % Input layer
    sequenceFoldingLayer('name','fold')                    % Fold time dimension for 2D convolution

    % CNN feature extraction
    convolution2dLayer([3,1],16,'Stride',[1,1],'name','conv1')  % Convolution layer
    batchNormalizationLayer('name','batchnorm1')                % Batch normalization
    reluLayer('name','relu1')                                   % ReLU activation

    % Pooling layer
    maxPooling2dLayer([2,1],'Stride',2,'Padding','same','name','maxpool')   % Max pooling layer
    % Unfold back to sequence
    sequenceUnfoldingLayer('name','unfold')       % Unfold after convolution
    flattenLayer('name','flatten')                % Flatten

    bilstmLayer(25,'Outputmode','last','name','hidden1') 
    selfAttentionLayer(1,2)          % Create a self-attention layer with 1 head and 2 keys/queries
    dropoutLayer(0.1,'name','dropout_1')        % Dropout layer (drop rate = 0.1)

    fullyConnectedLayer(1,'name','fullconnect')   % Fully connected layer
    regressionLayer('Name','output')    % Regression output layer
];

lgraph0 = layerGraph(layers0);
lgraph0 = connectLayers(lgraph0,'fold/miniBatchSize','unfold/miniBatchSize');

% Set the hyperparameters for training
options0 = trainingOptions('adam', ...              
    'MaxEpochs', 30, ...                            % Max number of training epochs
    'GradientThreshold', 1, ...                     % Gradient threshold
    'InitialLearnRate', 0.01, ...                   % Initial learning rate
    'L2Regularization', 0.001, ...                  % L2 regularization
    'ExecutionEnvironment', 'cpu',...               % Training environment
    'Verbose', 1, ...                               
    'Plots', 'none');                               
% Train the model
tic
net = trainNetwork(trainD,targetD',lgraph0,options0);
toc

% Prediction and evaluation
t_sim = net.predict(testD);  
analyzeNetwork(net);  % View model structure

% Reverse normalization
T_sim = mapminmax('reverse', t_sim, ps_output);

% Format conversion
T_sim = double(T_sim);
T_sim = T_sim';

% Optimize CNN-BiLSTM-Attention

disp(' ')
disp('Optimizing CNN-BiLSTM-Attention network:')

popsize=5;   % Population size
maxgen=10;   % Maximum generations
fobj = @(x)objectiveFunction(x,f_,trainD,targetD',testD,T_test,ps_output);
% Optimization settings
lb = [0.001 10 2  0.0001]; % Lower bounds: [learning rate, LSTM units, attention dim, L2]
ub = [0.01 50 50 0.001];   % Upper bounds
dim = length(lb);          % Dimension

[Best_score,Best_pos,curve]=SSO(popsize,maxgen,lb,ub,dim,fobj); % SSO optimization
setdemorandstream(pi);

figure
plot(curve,'r-','linewidth',2)
xlabel('Generation')
ylabel('MSE')
legend('Best Fitness')
title('Optimization Curve')

[~,optimize_T_sim] = objectiveFunction(Best_pos,f_,trainD,targetD',testD,T_test,ps_output);
setdemorandstream(pi);

str={'Ground Truth','CNN-BiLSTM-Attention','Optimized CNN-BiLSTM-Attention'};
figure('Units', 'pixels', ...
    'Position', [300 300 860 370]);
plot(T_test,'-','Color',[0.8500 0.3250 0.0980]) 
hold on
plot(T_sim,'-.','Color',[0.4940 0.1840 0.5560]) 
hold on
plot(optimize_T_sim,'-','Color',[0.4660 0.6740 0.1880])
legend(str)
set (gca,"FontSize",12,'LineWidth',1.2)
box off
legend Box off

test_y = T_test;
Test_all = [];

y_test_predict = T_sim;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];

y_test_predict = optimize_T_sim;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];

str={'Ground Truth','CNN-BiLSTM-Attention','Optimized CNN-BiLSTM-Attention'};
str1=str(2:end);
str2={'MAE','MAPE','MSE','RMSE','R2'};
data_out=array2table(Test_all);
data_out.Properties.VariableNames=str2;
data_out.Properties.RowNames=str1;
disp(data_out)

% Bar chart for MAE, MAPE, RMSE
color= [0.66669 0.1206 0.108;
        0.1339 0.7882 0.8588;
        0.1525 0.6645 0.1290;
        0.8549 0.9373 0.8275;   
        0.1551 0.2176 0.8627;
        0.7843 0.1412 0.1373;
        0.2000 0.9213 0.8176;
        0.5569 0.8118 0.7882;
        1.0000 0.5333 0.5176];

figure('Units', 'pixels', ...
    'Position', [300 300 660 375]);
plot_data_t=Test_all(:,[1,2,4])';
b=bar(plot_data_t,0.8);
hold on

for i = 1 : size(plot_data_t,2)
    x_data(:, i) = b(i).XEndPoints'; 
end

for i =1:size(plot_data_t,2)
    b(i).FaceColor = color(i,:);
    b(i).EdgeColor=[0.3353 0.3314 0.6431];
    b(i).LineWidth=1.2;
end

for i = 1 : size(plot_data_t,1)-1
    xilnk=(x_data(i, end)+ x_data(i+1, 1))/2;
    b1=xline(xilnk,'--','LineWidth',1.2);
    hold on
end 

ax=gca;
legend(b,str1,'Location','best')
ax.XTickLabels ={'MAE', 'MAPE', 'RMSE'};
set(gca,"FontSize",10,"LineWidth",1)
box off
legend box off

% 2D scatter
figure
plot_data_t1=Test_all(:,[1,5])';
MarkerType={'s','o','pentagram','^','v'};
for i = 1 : size(plot_data_t1,2)
   scatter(plot_data_t1(1,i),plot_data_t1(2,i),120,MarkerType{i},"filled")
   hold on
end
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off
legend(str1,'Location','best')
xlabel('MAE')
ylabel('R2')
grid on

% Radar chart
figure('Units', 'pixels', ...
    'Position', [150 150 520 500]);
Test_all1=Test_all./sum(Test_all);  % Normalize all metrics
Test_all1(:,end)=1-Test_all(:,end); % Reverse R2
RC=radarChart(Test_all1);
str3={'MAE','MAPE','MSE','RMSE','R2'};
RC.PropName=str3;
RC.ClassName=str1;
RC=RC.draw(); 
RC.legend();
RC.setBkg('FaceColor',[1,1,1])
RC.setRLabel('Color','none')
colorList=[78 101 155;
          181 86 29;
          184 168 207;
          231 188 198;
          253 207 158;
          239 164 132;
          182 118 108]./255;

for n=1:RC.ClassNum
    RC.setPatchN(n,'Color',colorList(n,:),'MarkerFaceColor',colorList(n,:))
end

% Polar compass chart
figure('Units', 'pixels', ...
    'Position', [150 150 920 600]);
t = tiledlayout('flow','TileSpacing','compact');
for i=1:length(Test_all(:,1))
nexttile
th1 = linspace(2*pi/length(Test_all(:,1))/2,2*pi-2*pi/length(Test_all(:,1))/2,length(Test_all(:,1)));
r1 = Test_all(:,i)';
[u1,v1] = pol2cart(th1,r1);
M=compass(u1,v1);
for j=1:length(Test_all(:,1))
    M(j).LineWidth = 2;
    M(j).Color = colorList(j,:);
end   
title(str2{i})
set(gca,"FontSize",10,"LineWidth",1)
end
legend(M,str1,"FontSize",10,"LineWidth",1,'Box','off','Location','southoutside')