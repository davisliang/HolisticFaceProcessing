%4/28/2015, 
%Davis Liang
function [weight, trainingError, steps, attempts] = completeCompositeTrainer(trainingData, targlen, testingData, activation, numberIterations)
%name identifying constants used in learning algorithm 
num_category = size(trainingData,2);   
num_trainImagesPerCategory = size(trainingData{1},2);
image_size = size(trainingData{1}(:,1),1); 
totalImages = num_category * num_trainImagesPerCategory;

inlen = image_size;

%Initializing network parameters and calculation constants
weight = (normrnd(0,1/sqrt(inlen+1),[targlen,inlen+1])); %random normalized weight distr.
learn = 0.01;
bias = [1];
a = 0.9;        %momentum
dwold = 0;      %old weight, temp holder
attempts = 1;
steps = 0;
run = 1;
Error = [];
TestPercentWrong = [];
testError = [];
epsilon = 0.001;
lambda = 0.1;
N = inlen;
%NOTES:
%1) How to use the Category data structure: 
%   Category{Category_number}(:,image_number);


%%
%Online Learning
while run
    steps = steps + 1;
    E = 0;
    for CategoryNum = randperm(num_category)       
        for imageNum = randperm(num_trainImagesPerCategory)
            %train
            input = trainingData{CategoryNum}(:,imageNum); %input is 40x1
            in = [bias; input];     %in is 41x1 column vector
            targ = zeros(num_category,1);       %5x1 column vector
            targ(CategoryNum) = 1;   %Turns on correct node for identity
       
            net = weight*in;        %weight is 5x41 matrix. net is 5x1 column vector
            
            if(strcmp(activation,'softmax'))
                out = exp(net)./sum(exp(net)); %softmax classifier. out is 5x1 col vector
                oprime = 1;     %slope of softmax classifier
            else
                out = [1./(1+exp(-net))];   %just use sigmoid, otherwise
                oprime = out.*(1-out);
            end
     
            delta = (targ - out).*oprime;    %error signal from output
            dw = learn*delta*in' + dwold*a;
            %if gradient descent is used for learning, the last term in the
            %cost function leads to a new term -lambda*weights in the
            %weight update. (for weight decay)

            weight = weight + dw; %old weight + change in weight
            dwold = dw; 
            E = E + 0.5*sum((targ-out).^2)/(totalImages);   

           %% GRADIENT CHECKING FOR SSE
%we want to make sure the actual partial of the error etc. is equal to the
%negative of delta*output. The actual partial derivative of error of
%calculated with the definition of sum squared error.
%numericalGradient = (f(x+epsilon)- f(x-epsilon))/(2*epsilon)
%epsilon = 0.001;

%to do gradient check one must first calculate each weight change by itself
%with the epsilon thing...

%{
weightPlus = weight;
weightMinus = weight;
gradApprox = zeros(5, 41);
ErrorPlus = 0;
ErrorMinus = 0;
for row = [1:5]
    for col = [1:41]
            weightPlus = weight;    %reset the weight matrix every time
            weightMinus = weight;
            
            weightPlus(row,col) = weight(row, col) + epsilon;   %add epsilon to right element
            weightMinus(row,col) = weight(row, col) - epsilon;
            
            netPlus = weightPlus*in;    %calculate the net input
            netMinus = weightMinus*in;
            
            outPlus = [1./(1+exp(-netPlus))];   %calculate the output
            outMinus = [1./(1+exp(-netMinus))];

            ErrorPlus = 0.5*sum((targ - outPlus).^2);   %find the SSE
            ErrorMinus = 0.5*sum((targ - outMinus).^2);
            
            gradApprox(row, col) = (ErrorPlus-ErrorMinus)/(2*epsilon);  %calculate each gradient approx
    end
end


fprintf('*******************************************');

fprintf('The numerical gradient is calculated as...');
gradApprox
fprintf('The gradient as shown by backpropogation is...');
gradBackProp = -delta*in'
fprintf('gradient difference');
gradDiff = gradApprox - gradBackProp

fprintf('*******************************************');
     %}   
        end    
    end
    
    
    if(steps>numberIterations) %resets network if it reaches poor local minima   
        E=0;   
    end
    
    %% Code for plotting Error and displaying learned weights  
    [tError, percentWrong, c] = completeCompositeTester(weight, testingData, activation);
    Error = [Error, E]; %Trainer SSE
    TestPercentWrong = [TestPercentWrong, percentWrong]; %Percent Wrong, SSE
    testError = [testError, tError]; 
    %close all;
    %plot(Error);
    if E < 0.001
        close all;
        fprintf('************************************************')
        fprintf('\nThe network learned weights:\n');
        weight
        figure;hold;
        plot(Error, 'r');
        plot(TestPercentWrong, 'g');
        plot(testError, 'b');
        legend('Training: SSE', 'Test: Percent Wrong', 'Test: SSE');
        fprintf('It took %i steps\n',steps);
        fprintf('It took %i attempts', attempts);
        fprintf('\nNOTE: First column of each weight matrix contains\n the bias weights\n');
        fprintf('************************************************\n')
        %Next, we classify and find percentage error
        trainingError = E;
        break;
    end  
    
end  %END OF WHILE LOOP, EXIT PROGRAM AFTER

%%



%