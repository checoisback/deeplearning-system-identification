clc
clear all
close all

X = table();
X_tmp = table();

for k = 1:100

    filename = strcat('patientData_pat',num2str(k),'.mat'); 
    load(filename);

    % set environment
    environment.scenario = 'single-meal';
    model.TS = 1; % physiological model sampling time. [minute]
    model.YTS = 5; % raw data sampling time. [minute]
    model.TID = minutes(data.Time(end)-data.Time(1)+minutes(5));  % from 1 to TID identify the model parameters. [min]
    model.TIDSTEPS = model.TID/model.TS;    % from 1 to TID identify the model parameters. [integration steps]
    model.TIDYSTEPS = model.TID/model.YTS;  % total identification simulation time [sample steps]
    mP.TS = model.TS;
    mP.BW = BW;    
    mP.tau = 8;
    mP.beta = 0;

    % prepare input data
    [bolus, basal, bolusDelayed, basalDelayed] = insulinSetupPF(data, model, mP);
    [cho, choDelayed] = mealSetupPF(data, model, mP, environment);

    meal = cho;
    total_ins = (bolus + basal);
    time = [0:12/(length(meal)-1):12]';
    timeOld = [0:12/(length(data.glucose)-1):12]';
    glucoseInterp = interp1(timeOld, data.glucose, time);

    X_tmp.time = [time];
    X_tmp.insulin = [total_ins];
    X_tmp.meal = [meal];
    X_tmp.glucose = [glucoseInterp];

    if k >1
        X = [X; X_tmp];
    else
        X = [X_tmp];
    end

    % timeData = data.Time(1):minutes(1):(data.Time(1) + minutes(length(cho)));

%     X(1:5,k) = [data.glucose(1); data.CHO(1); data.basal(1); data.bolus(1); BW];
%     Y(:,k) = data.glucose(2:end); 
    
%     y(:,k) = data.glucose;
%     x1(:,k) = data.basal;
%     x2(:,k) = data.bolus;
%     x3(:,k) = data.CHO;
    
    
    
end

Xtrain = X(1:57600,:);
Xtest = X(57601:end,:);

writetable(Xtrain, 'populationDataTrain.txt')
writetable(Xtest, 'populationDataTest.txt')

% writetable(array2table(y),'glucose.txt');
% writetable(array2table(x1),'basal.txt');
% writetable(array2table(x2),'bolus.txt');
% writetable(array2table(x3),'carbs.txt');

% Y = array2table(Y);
% X = array2table(X);

% writetable(Y,'output.txt');
% writetable(X,'input.txt');
