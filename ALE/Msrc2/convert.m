gt=dir('GroundTruth/*.jpg');
mkdir GroundTruth1
for i=1:length(gt)
    i
   fname=['GroundTruth/' gt(i).name];
    
   im=imread(fname);
   newim=imresize(im,0.3125,'nearest');
   imwrite(newim,['GroundTruth1/' gt(i).name]);
   
   
    
end

gt=dir('Images/*.jpg');
mkdir Images1
for i=1:length(gt)
    i
   fname=['Images/' gt(i).name];
    
   im=imread(fname);
   newim=imresize(im,0.3125,'bilinear');
   imwrite(newim,['Images1/' gt(i).name]);
   
   
    
    
end
