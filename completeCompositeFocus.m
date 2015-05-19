function [topFocusedGabor, botFocusedGabor] = focus(gaborMatrix, std_dev)
%4/22/2015, Davis Liang
%this is a script for top or bottom recognition
%this function takes in arguments of type matrix and does top saliency
%input image data

image_size = size(gaborMatrix); %gets the size of image as [#rows, #cols]
image_rows = image_size(1);
image_cols = image_size(2);

top_rows = image_rows/3; %change this

%BOTTOM FOCUS
top = 0.125 * ones(round(top_rows - 2*std_dev), image_cols);
bot = ones(round(image_rows - top_rows + 2*std_dev), image_cols);
total = [top; bot];
botFocusedGabor = total.*gaborMatrix;

%TOP FOCUS
top = ones(round(top_rows + 2*std_dev), image_cols);
bot = 0.125 * ones(round(image_rows - top_rows - 2*std_dev), image_cols);
total2 = [top; bot];
topFocusedGabor = total2.*gaborMatrix;
%create a matrix with 0.001 for top half and 1 for entire bot half


    



