clc
clear all
close all


for k = 1:100
    filename = strcat('patientData_pat',num2str(k),'.mat'); 
    load(filename);

    X(1:5,k) = [data.glucose(1); data.CHO(1); data.basal(1); data.bolus(1); BW];
    Y(:,k) = data.glucose(2:end); 
    
end

Y = array2table(Y);
X = array2table(X);

writetable(Y,'output.txt');
writetable(X,'input.txt');
