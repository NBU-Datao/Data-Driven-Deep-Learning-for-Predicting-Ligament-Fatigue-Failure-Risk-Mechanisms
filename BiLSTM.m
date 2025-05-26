clc;
clear 
close all
addpath('..\') %Load the previous directory

pwd
X = xlsread('Dataset.xlsx');

n_in = 5;  % Enter the data for the previous 5 moments
n_out = 1 ; % Set n_out to 1
or_dim = size(X,2) ;       % Record feature data dimensions
num_samples = 2000;  % Make 2000 samples.
scroll_window = 2;  %If it is equal to 1, the next data is taken from the second row. If it is equal to 2, the next data is taken from the third row.
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);


% Training set and test set division%

num_size = 0.8;                              % The proportion of training set to data set  
num_train_s = round(num_size * num_samples); % Number of training set samples  


P_train = res(1: num_train_s,1);
P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
T_train = res(1: num_train_s,2);
T_train = cell2mat(T_train)';

P_test = res(num_train_s+1: end,1);
P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
T_test = res(num_train_s+1: end,2);
T_test = cell2mat(T_test)';


%  Data Normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

vp_train = reshape(p_train,n_in,or_dim,num_train_s);
vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);

vt_train = t_train;
vt_test = t_test;

f_ = [size(vp_train,1) size(vp_train,2)];
outdim = n_out;

%  Creating a BiLSTM Network，
layers = [ ...
    sequenceInputLayer(f_)              % Input layer
    flattenLayer
    bilstmLayer(25)                      
    reluLayer                           
    fullyConnectedLayer(outdim)         % regression layer
    regressionLayer];

%  Parameter settings
options = trainingOptions('adam', ...                 
    'MaxEpochs', 150, ...                            
    'GradientThreshold', 1, ...                       
    'InitialLearnRate', 0.01, ...        
    'LearnRateSchedule', 'piecewise', ...             
    'LearnRateDropPeriod', 70, ...                   
    'LearnRateDropFactor',0.1, ...                    
    'L2Regularization', 0.001, ...         
    'ExecutionEnvironment', 'cpu',...                 
    'Verbose', 1, ...                                 
    'Plots', 'none');                    

%  Training
tic
net = trainNetwork(vp_train, vt_train, layers, options);
toc
%analyzeNetwork(net);% 
%  Predict
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%  Data denormalization
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);


% 指标计算
disp('…………Training set error indicator…………')
[mae1,rmse1,mape1,error1]=calc_error(T_train,T_sim1);
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_train);
hold on
plot(T_sim1)
legend('real value','predicted value')
title('BiLSTM Comparison of prediction results of training set')
xlabel('sample point')
ylabel('Results')


disp('…………Test set error indicator…………')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')


figure('Position',[200,300,600,200])
plot(T_test2);
hold on
plot(T_sim2')
legend('real value','predicted value')
title('BILSTM Comparison of prediction results of training set')
xlabel('sample point')
ylabel('Results')

figure('Position',[200,300,600,200])
plot(T_sim2'-T_test2)
title('BILSTM Error curve graph')
xlabel('sample point')
ylabel('Results')
