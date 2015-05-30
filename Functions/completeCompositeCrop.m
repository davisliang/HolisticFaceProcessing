%Davis Liang, April 14, 2015.
%GURU IMAGE CROPPER
%%
%Make sure you are in the POFA folder...

%User Input To Here.
cropRect = [0,100,300,300];
cd('~/Desktop/POFA-Faces-Aligned-2-00');
loc = ['./aligned'];

%Code here.
cd(loc);
FileDirectory = dir(pwd); %to presesnt working directory
numImages = 0; %To count number of images and return

for eachImage = 3:length(FileDirectory) %first two indices are . and ..
    numImages = numImages + 1;
    image = imread(FileDirectory(eachImage).name);   
    [croppedImage] = imcrop(image, cropRect);
    croppedImagePath = strcat('../cropped_aligned_bottom/', 'image', num2str(numImages), '.png');
    imwrite(croppedImage, croppedImagePath);
    
end
