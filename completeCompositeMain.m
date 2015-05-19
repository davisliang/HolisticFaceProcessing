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
oneLayer = true;
numberIterations = 500;
activation = 'sigmoid'; %'sigmoid';

preprocessLoc = 'C:/Users/davis_000/Desktop/Composite Face Code/EmotionPOFA';    %image location
matFileLoc = 'C:/Users/davis_000/Desktop/Composite Face Code'; %matFileLocation
matFile = 'PrepDataPOFA_Tr10_Te4.mat'; %matfile name
imPerFolder = 14;
numTest =4; %14 in total
targetLength = 7; 


if(preprocess)
    PreprocessDataSetPlus(topFocus, fullFocus, preprocessLoc, showFilteredHalfFace, imPerFolder, numTest);
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
    [percentWrong, numberWrong] = completeCompositeTester(learnedWeight, testingData, activation)
else
    [whi, woh, trainingError, trainingSteps, trainingAttempts]  = completeCompositeTrainerTwoLayer(trainingData, targetLength, testingData, activation, numberIterations) %output weight matrix with results too
    [percentWrong, numberWrong] = completeCompositeTesterTwoLayer(whi, woh, testingData, activation)
end


