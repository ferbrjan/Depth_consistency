function counts = softmax_count(distances)
    minDist = 0;
    maxDist = 25.5;
    numBins = ceil((maxDist - minDist) / 0.1) + 1;
    
    % Initialize the histogram counts vector
    counts = zeros(1, numBins);
    
    % Compute the counts for each distance
    for i = 1:length(distances)
        dist = distances(i);
        bin = floor((dist - minDist) / 0.1) + 1;
        if bin > 256
            bin = 256;
        end
        counts(bin) = counts(bin) + 1;
    end
    counts = counts / length(distances);
end