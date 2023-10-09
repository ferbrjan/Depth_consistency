load("00020.mat")

%% 
ids_correct = find(categories==1);
ids_wrong = find(categories==3);

cellsz = cellfun(@length,GTs);
ids_correct_size =  find(cellsz==18);

[val_correct,~]=intersect(ids_correct,ids_correct_size);
[val_wrong,~]=intersect(ids_wrong,ids_correct_size);

%data = {};
softmax_out = [];
means = [];
stds = [];
length(val_correct)
for i = 1:length(val_correct)
    i
    %Take one random outlier sample for each non-outlier sample
    id = randi(length(val_wrong),1);
    GT_id = val_correct(i);
    outlier_id = val_wrong(id);

    %Get all GT values needed
    GT = GTs{GT_id};
    GT_mean =  avgs(GT_id);
    GT_median = medians(GT_id);
    GT_stdev =  stdevs(GT_id);
    %visualize(GT)

    %Get all outlier values needed
    outlier = GTs{outlier_id};
    outlier_mean =  avgs(outlier_id);
    outlier_median = medians(outlier_id);
    outlier_stdev =  stdevs(outlier_id);
    %visualize(outlier)

    %Allign outlier sample to GT
    mean_diff = GT_mean - outlier_mean;
    median_diff = GT_median - outlier_median;
    outlier_shifted = outlier+(mean_diff+median_diff)/2;
    outlier_shifted = outlier_shifted(outlier_shifted>=0);
    outlier_shifted = outlier_shifted(outlier_shifted<=15);
    %visualize(outlier_shifted)

    %Join them
    out =  [GT,outlier_shifted];
    %visualize(out)

    %Subsample
    cnt = randi(2,1);
    while length(out)>18
        if cnt>length(out)
            break
        end
        out(cnt)=[];
        cnt = cnt + 1;
    end
    %visualize(out)

    %Save
    %data{end+1}.GT = GT;
    %data{end}.GT_softmax =  softmax_count(GT);
    %data{end}.GT_mean = GT_mean;
    %data{end}.stdev = GT_stdev;
    %data{end}.out = out;
    %data{end}.out_softmax =  softmax_count(out);

    
    softmax_out = [softmax_out;softmax_count(out)];
    means =  [means,GT_mean];
    stds = [stds,GT_stdev];
end

%%
save("00020_ready.mat","softmax_out","stds","means")
