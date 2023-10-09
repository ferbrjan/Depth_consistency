function visualize(set)

    figure("Name","Distances on ray")
    hAxes = axes('NextPlot','add',...          
         'DataAspectRatio',[1 1 1],...  
         'XLim',[0,10],...               
         'YLim',[0 eps],...             
         'Color','none');  
    plot(set,0,"Marker","*","color","r",'MarkerSize',10);  


end