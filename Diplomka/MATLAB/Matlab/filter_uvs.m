function [uv] = filter_uvs(uvl, rgb)
    uvl(:,uvl(3,:)<0) = [];
    uv = h2a(uvl);
    uv(:,uv(1,:)<0) = [];
    uv(:,uv(2,:)<0) = [];
    uv(:,uv(1,:)>size(rgb,2)) = [];
    uv(:,uv(2,:)>size(rgb,1)) = [];
end

