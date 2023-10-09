close all;
clear;

[cameras, images, points3D] = read_model('./colmap_unit_test/in');
saveColmap( './colmap_unit_test/out', cameras, images, points3D ); 
