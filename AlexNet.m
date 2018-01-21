% De acordo com exemplo do matlab: https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html
%(elaborado à mao e de acordo com treino ja elaborado)

clear all;
close all;

% armazenamento das imagens segundo uma estrutura imagaDatastore (aqui vao
% ser percorridas tanto a PastaBOCAS como a PastaFACES e a PastaOLHOS
% segundo as categorias .png1,.png2,etc
imds = imageDatastore('C:\Users\danis\Documents\Base de dados\projecto 2\testado\PastaFACES','FileExtensions',{'.png1','.png2','.png3','.png4','.png5','.png6','.png7'});
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
labels=categorical(zeros(327,1));

%label da imagem
for i=1:327
    label=str2double(imds.Files{i,1}(end));
    labels(i)=categorical(label);
end

imds.Labels=labels;
%% train and test set

[trainingset,testset] = splitEachLabel(imds,0.7,'randomized'); % treino com 70% dos dados e teste com 30%
%% Layers for training CNN

convlayer96 = convolution2dLayer(11,96,'Stride',4, 'BiasL2Factor' ,1); % Stride: deslocamento do filtro.
convlayer256 = convolution2dLayer(5,256,'Stride',1,'Padding',2,'BiasL2Factor' ,1); % Stride: deslocamento do filtro.
convlayer384 = convolution2dLayer(3,384,'Stride',1,'Padding',1,'BiasL2Factor' ,1); % Stride: deslocamento do filtro.
convlayer256_v2 = convolution2dLayer(5,256,'Stride',1,'Padding',1,'BiasL2Factor' ,1); % Stride: deslocamento do filtro.

localnormlayer = crossChannelNormalizationLayer(5);
maxpoollayer = maxPooling2dLayer(3,'Stride',2);
trans_layer = reluLayer();
%% combinaçao das layers num vector
layers = [imageInputLayer([48 48 1]); maxpoollayer;convlayer256;maxpoollayer;convlayer384;maxpoollayer;convlayer256_v2; maxpoollayer;fullyConnectedLayer(4096); fullyConnectedLayer(7); softmaxLayer();classificationLayer()];

% layers = [imageInputLayer([224 224 1]); convlayer96;trans_layer;localnormlayer;maxpoollayer;convlayer256;trans_layer;localnormlayer;maxpoollayer;convlayer384;trans_layer;convlayer384;trans_layer;convlayer256_v2;trans_layer;maxpoollayer;fullyConnectedLayer(4096);trans_layer;fullyConnectedLayer(4096,'Name','fc7');trans_layer;fullyConnectedLayer(7); softmaxLayer();classificationLayer()];
% layers = [imageInputLayer([64 64 3]);
%           convolution2dLayer(5,20);
%           reluLayer();
%           maxPooling2dLayer(2,'Stride',2);
%           fullyConnectedLayer(7);
%           softmaxLayer();
%           classificationLayer()];
%% Convnet training
options = trainingOptions('sgdm','InitialLearnRate',0.2,'LearnRateSchedule','none','MiniBatchSize',30);
rng('default')
convnet = trainNetwork(trainingset,layers,options);
%% activaçao das features relevantes para posterior classificaçao
featureLayer = 'fc7'; %layer anterior à layer de classificaçao
trainingFeatures = activations(convnet, trainingset, 22,'MiniBatchSize', 300);%, 'OutputAs', 'columns');
%% realizaçao de classificador svm de treino
svm = fitcecoc(trainingFeatures,trainingset.Labels);
%% activaçao das features relevantes para teste e sua classificaçao
testFeatures = activations(convnet,testset,22);
testPredictions = predict(svm,testFeatures);
%% matriz confusao
C = confusionmatStats(testset.Labels,testPredictions)
