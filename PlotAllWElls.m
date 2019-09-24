clear;
close all;

%open data file
data=xlsread("\\RFIbackup.che.caltech.edu\Group_Files\jrolando\dRTI\20171208_ClinicalSamples_IndividualChips\C0VERI_U42C.csv");
%plot data and annotate
plot(data(:,1), data(:,2:end));
ylim([0 2000])
xlim([0 120])
xlabel('Frame number (2 per minute)');
ylabel('RFU');
title('C0VERI U42C');
%save plot as figure and PNG
print('\\RFIbackup.che.caltech.edu\Group_Files\jrolando\dRTI\20171208_ClinicalSamples_IndividualChips\Plots\C0VERI_U42C', '-dpng')