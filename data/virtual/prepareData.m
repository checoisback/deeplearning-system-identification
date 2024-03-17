clc
clear all
close all

load dataset

patientData = dataset{1,1};

for k = 1:floor(height(patientData)/288)
    DataTmp = patientData(288*(k-1)+1:288*k,:);
    y(:,k) = DataTmp.CGM;
    x1(:,k) = DataTmp.Insulin_basal;
    x2(:,k) = DataTmp.Insulin_bolus;
    x3(:,k) = DataTmp.Carbohydrates;
end
y = array2table(y);
x1 = array2table(x1);
x2 = array2table(x2);
x3 = array2table(x3);
writetable(y,'glucose.txt');
writetable(x1,'basal.txt');
writetable(x2,'bolus.txt');
writetable(x3,'carbs.txt');
