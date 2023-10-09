function [name2id] = images_name2id(images)
    name2id = containers.Map('KeyType','char','ValueType','double');
    c_images = images.values();
    for i = 1:size(c_images,2) 
        name2id(strtrim(c_images{i}.name)) = c_images{i}.image_id;
    end
end

