clear all,clc,close all

unzip("imgdata.zip")
imds = imageDatastore("imgdata", ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

numTrainFiles = 30;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomized");

classNames = categories(imdsTrain.Labels)

inputSize = [32 32 1];
numClasses = 4;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,5)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer];

options = trainingOptions("sgdm", ...
    MaxEpochs=4, ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=30, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

net = trainnet(imdsTrain,layers,"crossentropy",options);