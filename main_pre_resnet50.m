clear; close all; clc;
cd("Utils");
% load("Workspace\featuresExtractedResnet50.mat")

%% prepara��o
fprintf("-> Preparando os dados\n");
tic

% rootFolder = 'E:\Pontificia Universidade Catolica de Goias\9_periodo\Vis�o Computacional\N2\Maio\Train';
% categories = {'aviao','automovel','passaro','gato','veado','cachorro','sapo','cavalo','navio','caminhao'};
% trainingSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
% trainingSet.ReadFcn = @readFunctionTrain;
% 
% testFolder = 'E:\Pontificia Universidade Catolica de Goias\9_periodo\Vis�o Computacional\N2\Maio\machine-learning-cifar-10-20201\Test\Test';
% testSet = imageDatastore(fullfile(testFolder));
% testSet.Files = natsortfiles(testSet.Files);
% testSet.ReadFcn = @readFunctionTrain;

rootFolder = 'E:\Pontificia Universidade Catolica de Goias\9_periodo\Vis�o Computacional\N2\Maio\Train';
categories = {'aviao','automovel','passaro','gato','veado','cachorro','sapo','cavalo','navio','caminhao'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

[trainingSet, testSet] = splitEachLabel(imds, 0.8);
toc
%% defini��o da rede
convnet = resnet50;

%% extra��o de features
fprintf("\n-> Realizando extra��o de feactures\n");
tic

featureLayer = 'fc1000';
trainingFeatures = activations(convnet, trainingSet, featureLayer,...
    'OutputAs', 'rows');

toc
%% treinamento modelo
fprintf("\n-> Realizando treinamento do modelo\n");
tic

t = templateSVM(...
    'KernelFunction', 'gaussian',...
    'KernelScale', 74.115,...
    'BoxConstraint', 0.0016294,...
    'Standardize', true);
classifier = fitcecoc(trainingFeatures, trainingSet.Labels,...
    'Coding', 'onevsall',...
    'Options', statset('UseParallel',true));

% classifier = fitcecoc(trainingFeatures, trainingSet.Labels,...
%     'OptimizeHyperparameters','all',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus',...
%     'UseParallel',true,...
%     'ShowPlots',false,...
%     'Verbose',1));

toc
%% extra��o de features teste
fprintf("\n-> Realizando extra��o de feactures teste\n");
tic

testFeatures = activations(convnet, testSet, featureLayer, 'OutputAs', 'rows');

toc
%% testando modelo
fprintf("\n-> Realizando predi��o\n");
tic

predictedLabels = predict(classifier, testFeatures);

toc
%% verificando acur�cia do modelo
fprintf("\n-> Verificando acur�cia do modelo\n");
tic

confMat = confusionmat(testSet.Labels, predictedLabels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

toc
%% gerando csv de resultados
% T = table([1:10000]',predictedLabels);
% T.Properties.VariableNames = {'Id','Label'};
% writetable(T,'Resultados\resultado2Gaussian.csv','QuoteStrings',false);

%% Finalizar execu��o
cd("..\")
%% functions
function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);
I = imresize(I, [224 224]);
end