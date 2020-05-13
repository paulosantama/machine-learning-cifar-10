clear; close all; clc;
% load("Workspace\featuresExtractedTrainTest.mat")

%% preparação
fprintf("-> Preparando os dados\n");
tic

rootFolder = 'E:\Pontificia Universidade Catolica de Goias\9_periodo\Visão Computacional\N2\Maio\Train';
categories = {'0','1','2','3','4','5','6','7','8','9'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

[trainingSet, testSet] = splitEachLabel(imds, 0.8);

toc
%% definição da rede
convnet = vgg16;

%% extração de feactures
fprintf("\n-> Realizando extração de feactures\n");
tic

featureLayer = 'fc6';
gpuDevice(1);
trainingFeatures = activations(convnet, trainingSet, featureLayer,...
    'MiniBatchSize',25,...
    'OutputAs', 'channels');

toc
%% treinamento modelo
fprintf("\n-> Realizando treinamento do modelo\n");
tic

% classifier = fitcnb(double(trainingFeatures), trainingSet.Labels);
% classifier = fitcknn(double(trainingFeatures), trainingSet.Labels);

t = templateSVM(...
    'BoxConstraint', 303.96,...
    'KernelScale', 230.38);
classifier = fitcecoc(double(trainingFeatures), trainingSet.Labels,...
    'Learners', t,...
    'Options', statset('UseParallel',true));

toc
%% extração de feactures teste
fprintf("\n-> Realizando extração de feactures teste\n");
tic

testFeatures = activations(convnet, testSet, featureLayer, 'OutputAs', 'channels');

toc
%% testando modelo
fprintf("\n-> Realizando predição\n");
tic

predictedLabels = predict(classifier, testFeatures);

toc
%% verificando acurácia do modelo
fprintf("\n-> Verificando acurácia do modelo\n");
tic

confMat = confusionmat(testSet.Labels, predictedLabels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

toc
%% functions
function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);
I = imresize(I, [227 227]);
end