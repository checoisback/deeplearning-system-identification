clc
clear all
close all

load dataset

for k = 1:size(dataset)
    patientData = dataset{k,1};
    y(:,k) = patientData.CGM;
    x1(:,k) = patientData.Insulin_basal;
    x2(:,k) = patientData.Insulin_bolus;
    x3(:,k) =patientData.Carbohydrates;
end
y = array2table(y);
x1 = array2table(x1);
x2 = array2table(x2);
x3 = array2table(x3);
writetable(y,'glucose.txt');
writetable(x1,'basal.txt');
writetable(x2,'bolus.txt');
writetable(x3,'carbs.txt');
