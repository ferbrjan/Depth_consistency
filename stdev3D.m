function deviation = stdev3D(points)
        
    stdev = std(points,0,2);
    deviation = sum(stdev);


end