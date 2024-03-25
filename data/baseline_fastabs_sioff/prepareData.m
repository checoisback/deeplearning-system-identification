clc
clear all
close all


for k = 1:100
    filename = strcat('patientData_pat',num2str(k),'.mat'); 
    load(filename);

%     X(1:5,k) = [data.glucose(1); data.CHO(1); data.basal(1); data.bolus(1); BW];
%     Y(:,k) = data.glucose(2:end); 
    
    y(:,k) = data.glucose;
    x1(:,k) = data.basal;
    x2(:,k) = data.bolus;
    x3(:,k) = data.CHO;
    
    
    
end
writetable(array2table(y),'glucose.txt');
writetable(array2table(x1),'basal.txt');
writetable(array2table(x2),'bolus.txt');
writetable(array2table(x3),'carbs.txt');

% Y = array2table(Y);
% X = array2table(X);

% writetable(Y,'output.txt');
% writetable(X,'input.txt');
