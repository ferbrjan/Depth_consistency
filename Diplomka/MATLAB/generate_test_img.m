% WARNING THIS TAKES A S***LOAD OF TIME AND STORAGE!!

load("00180.mat")
im_w = 1920;
im_h = 1440;
%% 
data = cellfun(@softmax_count,GTs,'uniformoutput',false);
%%
save("00180_test.mat","data",'-v7.3')