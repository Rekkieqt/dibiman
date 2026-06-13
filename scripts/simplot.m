clear all
close all
clc

args = argv();
arm = args{1};
arm = "right"
path_meas = ["../logs/" arm "_arm.csv"];
path_ref = ["../logs/" arm "_arm_ref.csv"];

logfile = csvread(path_meas);
logfile = logfile(2:end,:);

reffile = csvread(path_ref);


figure
hold on
for i=1 : length(logfile(1,:))
  plot(logfile(:,i))


for j=1 : length(reffile)
  ref_j = ones(length(logfile(:,1))) * reffile(j)
  plot(ref_j, '--')
