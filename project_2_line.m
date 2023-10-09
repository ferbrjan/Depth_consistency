function [projectedPoints,distances] = project_2_line(direction, translation, points)

[numPoints,~] = size(points);
projectedPoints = zeros(numPoints, 3);
distances = zeros(numPoints, 1);
for i = 1:numPoints
    point = points(i, :);
    
    % Calculate the projection
    t = translation;
    d = direction / norm(direction); % Normalize direction vector
    q = t + dot(point - t, d) * d;
    
    projectedPoints(i, :) = q;

    distances(i) = norm(q - t);
end

end