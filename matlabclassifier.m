clear all,clc,close all

unzip("imgdata.zip")
imds = imageDatastore("isolated", ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

numTrainFiles = 70;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomized");

classNames = categories(imdsTrain.Labels);

inputSize = [16 16 1];
numClasses = 2;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(10,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer];

options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=30, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

net = trainnet(imdsTrain,layers,"crossentropy",options);