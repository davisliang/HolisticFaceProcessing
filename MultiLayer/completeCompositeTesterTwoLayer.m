%4/22/2015, Davis Liang
function [percentWrong, Error, numberWrong] = completeCompositeTesterTwoLayer(whi, woh, testingData, activation)
numberWrong = 0;
Error = 0;
bias = [1];
%name identifying constants used in learning algorithm 
num_category = size(testingData,2);
num_trainImagesPerCategory = size(testingData{1},2);
totalImages = num_category * num_trainImagesPerCategory;


%testing code here. Essentially, take the testingData, multiply with the
%weight matrix and then find the error. this is 1 iteration!
for numCategories = (1:num_category)           
        for imageNum = (1:num_trainImagesPerCategory)
            testInput = [bias;testingData{numCategories}(:,imageNum)];
            
            targ = zeros(num_category,1);      
            targ(numCategories) = 1;   

            %hidden layer
            neti = [(whi(:,:)*testInput)];  %net input to either hidden unit     
            hout = [1./(1+exp(-neti))];
            
            %output layer
            h_layer = [bias; hout];         %complete hidden layer
            neto = woh*[h_layer];           %net input to each output unit                  
            
            if(strcmp(activation,'sigmoid'))
                out = (1./(1+exp(-neto)));
            else
                out = exp(neto)./sum(exp(neto));
            end




            
            %number wrong code here
             %number wrong code here
            [maxValue maxIndex] = max(out);
            if maxIndex ~= numCategories
                numberWrong = numberWrong + 1;
            end
            Error = Error + 0.5*sum((targ-out).^2)/totalImages;
            %}
        end
end
percentWrong = numberWrong/totalImages;