%4/28/2015, Davis Liang
function [percentWrong, numberWrong] = completeCompositeTester(weight, testingData, activation)
numberWrong = 0;
testError = 0;
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

            net = weight*testInput;
            
            if(strcmp(activation,'softmax'))
                out = exp(net)./sum(exp(net)); 
            else
                out = [1./(1+exp(-net))];
            end

            
            fprintf('**********TESTING*********');
            targ
            out
            
            %number wrong code here
            [maxValue maxIndex] = max(out);
            if maxIndex ~= numCategories
                numberWrong = numberWrong + 1;
                fprintf('wrong\n');
            else
                fprintf('correct\n');
            end

            %}
            
        end
end
percentWrong = numberWrong/totalImages;

