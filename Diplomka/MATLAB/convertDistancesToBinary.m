function binaryVector = convertDistancesToBinary(distances, step, maxDistance)
    minDistance = 0;
    %maxDistance = max(distances);
    numSteps = floor((maxDistance - minDistance) / step) + 1;

    binaryVector = zeros(1, numSteps);

    for i = 1:length(distances)
        index = floor((distances(i) - minDistance) / step) + 1;
        binaryVector(index) = 1;
    end
end