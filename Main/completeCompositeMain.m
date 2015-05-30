%5/8/2015, Davis Liang, Version 3.1 
%This algorithm will learn a one-layer (input to output) representation of
%the preprocessed data. We will use Xentropy error and a softmax function
%to calculate this representation.
close all;
%before pca, zscore the gabor filters... take one gabor filter z score it.
%0 degree z score that way over whole image. (subtract mean etc. for each
%filter for all images.
%eigenvalue demo on resource in gary's page halfway down

preprocess = true;
fullFocus = true;
topFocus = false; %toggles between bot and top focus
showFilteredHalfFace = false;
oneLayer = false;
numberIterations = 500;
activation = 'sigmoid'; %'sigmoid';
numhid = 20;

preprocessLoc = 'C:/Users/davis_000/Desktop/Composite Face Code/SampleDataSet';    %image location
matFileLoc = 'C:/Users/davis_000/Desktop/Composite Face Code'; %matFileLocation
matFile = 'PrepDataID_Tr13_Te3.mat'; %matfile name

numTrain = 8
numTest = 8 %14 in total

imPerFolder = numTrain + numTest;
targetLength = 5; 



if(preprocess)
    PreprocessDataSetPlus(topFocus, fullFocus, preprocessLoc, showFilteredHalfFace, imPerFolder, numTest, matFile);
end

cd(matFileLoc);
matFile = load(matFile);
data = matFile.preprocessedData;

close all;
%recall that data holds 3 things: 
%1) name, 'Faces'
%2) trainingSet, a 1x5 cell of 40x12 doubles.
%   a) 1x5 cell due to each person
%   b) 40x12 because 8 principle components for 5 scales (PCA drops 'orientation') and 12 images.
%3) testSet,     a 1x5 cell of 40x4  doubles.
%   a) 1x5 cell due to each person
%   b) 40x4 because 8 Principle components for 5 scales (PCA drops 'orientation') and 4 images.

trainingData = data.trainingSet;
testingData = data.testSet;

%person{1}; %first person
%person{1}(:,1); %first person's first image
%person{1}(1:8, 1) %first person, first image, first scale of gabor (8 PCAs). 
%aka at this scale which orientations, encoded, we strongest?

%person{person_number}(:,image_number);

%need a sigmoid network of 40 inputs and 5 outputs for emotion. create
%separate file to store emotive information

if(oneLayer == true)
    [learnedWeight, trainingError, trainingSteps, trainingAttempts]  = completeCompositeTrainer(trainingData, targetLength, testingData, activation, numberIterations) %output weight matrix with results too
    [testError, percentWrong, numberWrong] = completeCompositeTester(learnedWeight, testingData, activation)
else
    [whi, woh, trainingError, trainingSteps, trainingAttempts]  = completeCompositeTrainerTwoLayer(trainingData, targetLength, testingData, activation, numberIterations, numhid) %output weight matrix with results too
    [percentWrong, testError, numberWrong] = completeCompositeTesterTwoLayer(whi, woh, testingData, activation)
end


