% De acordo com exemplo do matlab: https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html

clear all;
close all;
%% 1)
% Location of pre-trained "AlexNet"
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
% Store CNN model in a temporary folder
cnnMatFile = fullfile(tempdir, 'imagenet-caffe-alex.mat');
% pre defined Alex convnet
if ~exist(cnnMatFile, 'file') % download only once
    disp('Downloading pre-trained CNN model...');
    websave(cnnMatFile, cnnURL);
end

%
% Load MatConvNet network into a SeriesNetwork
matlab_convnet = helperImportMatConvNet(cnnMatFile);

%% labelling
labels=categorical(zeros(327,1));
%% Create image datastore with training and testing sets
imds = imageDatastore('C:\Users\danis\Documents\Base de dados\projecto 2\testado\PastaFACES','FileExtensions',{'.png1','.png2','.png3','.png4','.png5','.png6','.png7'});

imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

for i=1:327
    label=str2double(imds.Files{i,1}(end));
    labels(i)=categorical(label);
end

imds.Labels=labels;
%% train and test set

[trainingset,testset] = splitEachLabel(imds,0.7,'randomized'); % treino com 70% dos dados e teste com 30%

%%
featureLayer = 'fc7'; %layer anterior à layer de classificaçao
trainingFeatures = activations(matlab_convnet, trainingset, featureLayer,'MiniBatchSize', 32, 'OutputAs', 'columns');

%%
% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.

classifier = fitcecoc(trainingFeatures, trainingset.Labels,'Learners', 'Linear','Coding', 'onevsall', 'ObservationsIn', 'columns');

%%
% Extract test features using the CNN
testFeatures = activations(matlab_convnet, testset, featureLayer, 'MiniBatchSize',32);
% 
% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

%% Tabulate the results using a confusion matrix.
confMat = confusionmat(testset.Labels, predictedLabels)