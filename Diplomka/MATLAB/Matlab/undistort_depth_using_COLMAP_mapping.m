output_path = './undistorted/depth';
h = 1440;
w = 1920;

% get images names
depth_image_names = dir('depth/*.mat');

% load the mapping function 
M = readmatrix('./undistorted/undistortion_mapping.txt');
M = M + 1;
orig_x = zeros(h,w);
orig_y = zeros(h,w);
for i = 1:size(M,1)
    orig_x(M(i,2),M(i,1)) = M(i,3);
    orig_y(M(i,2),M(i,1)) = M(i,4);
end

% apply the mapping/undistortion
for i = 1:size(depth_image_names,1)
    fprintf('Undistort %.0f/%.0f \n',i,size(depth_image_names,1))
    load(fullfile('depth',depth_image_names(i).name));
    fit_depth = imresize(depth, [h,w]);
    undist_depth = zeros(h,w);
    for j = 1:h
        for k = 1:w 
            y = orig_y(j,k);
            x = orig_x(j,k);
            if x == y && x == 0
               continue; 
            end
            if floor(x) == ceil(x) && floor(y) == ceil(y)
                undist_depth(j,k) = fit_depth(round(y),round(x));
                continue;
            end
            if floor(x) == ceil(x)
                undist_depth(j,k) = [fit_depth(ceil(y),round(x)) fit_depth(floor(y),round(x))] * ...
                    [y - floor(y); ceil(y) - y];
                continue;
            end
            if floor(y) == ceil(y)
                undist_depth(j,k) = [ceil(x) - x, x - floor(x)] * ...
                    [fit_depth(round(y),floor(x)); fit_depth(round(y),ceil(x))];
                continue;
            end
            if floor(x) >= 1 && ceil(x) <= w && floor(y) >= 1 && ceil(y) <= h 
                Q = [fit_depth(ceil(y),floor(x)) fit_depth(floor(y),floor(x)); ...
                    fit_depth(ceil(y),ceil(x)) fit_depth(floor(y),ceil(x))];
                undist_depth(j,k) = [ceil(x) - x, x - floor(x)] * Q * [y - floor(y); ceil(y) - y];
                continue;
            end
            undist_depth(j,k) = NaN;
        end
    end
    depth = undist_depth;
%     depth2show = double(depth / max(depth(:)));
%     subfig(1,1,1); imshow(depth2show);
    save(fullfile(output_path,depth_image_names(i).name),'depth')
end