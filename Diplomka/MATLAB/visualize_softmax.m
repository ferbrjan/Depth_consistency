function visualize_softmax(softmax)

    figure("Name","Softmax")
    bar(softmax)
    ylabel("probability [%]")
    xlabel("distance [cm]")

end