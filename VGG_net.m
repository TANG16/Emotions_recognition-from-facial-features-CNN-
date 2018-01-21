% (a preencher) VGG NET complexa de acordo com http://cs231n.github.io/convolutional-networks/

clear all;
close all;

imds = imageDatastore('C:\Users\danis\Documents\Base de dados\projecto 2\testado\PastaFACES','FileExtensions',{'.png1','.png2','.png3','.png4','.png5','.png6','.png7'});
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
labels=categorical(zeros(327,1));


for i=1:327
    label=str2double(imds.Files{i,1}(end));
    labels(i)=categorical(label);
end

imds.Labels=labels;
%% train and test set

[trainingset,testset] = splitEachLabel(imds,0.7,'randomized'); % treino com 70% dos dados e teste com 30%
%% simple example CNN
    
    % Convnet construction
    
    convlayer64 = convolution2dLayer(3,32,'Stride',1,'Padding',1); % Stride: deslocamento do filtro.
    convlayer128 = convolution2dLayer(3,64,'Stride',1,'Padding',1); % Stride: deslocamento do filtro.
    convlayer256 = convolution2dLayer(3,128,'Stride',1,'Padding',1); % Stride: deslocamento do filtro.
    convlayer512 = convolution2dLayer(3,256,'Stride',1,'Padding',1); % Stride: deslocamento do filtro.

    maxpoollayer = maxPooling2dLayer(2,'Stride',2);
    
    trans_layer = reluLayer('Name','relu1');
    
%     layers = [imageInputLayer([224 224 3],'Normalization','none');convlayer64; trans_layer;convlayer64; trans_layer; maxpoollayer;convlayer128; trans_layer;convlayer128; trans_layer; maxpoollayer;convlayer256; trans_layer;convlayer256; trans_layer;convlayer256; trans_layer;maxpoollayer;convlayer512; trans_layer;convlayer512; trans_layer;convlayer512; trans_layer;maxpoollayer;convlayer512; trans_layer;convlayer512; trans_layer;convlayer512; trans_layer; fullyConnectedLayer(4096);fullyConnectedLayer(4096);fullyConnectedLayer(7,'Name','fc7'); softmaxLayer();classificationLayer()];
    layers = [imageInputLayer([64 64 1]);convlayer64;convlayer64; maxpoollayer;convlayer128; convlayer128; maxpoollayer;convlayer256; convlayer256;convlayer256;  maxpoollayer;fullyConnectedLayer(4096);fullyConnectedLayer(4096); fullyConnectedLayer(7,'Name','fc7'); softmaxLayer();classificationLayer()];

%%
    % training convnet
    options = trainingOptions('sgdm','InitialLearnRate',0.05,'LearnRateSchedule','piecewise','MiniBatchSize',128);
    rng('default')
    convnet = trainNetwork(trainingset,layers,options);

%%
featureLayer = 'fc7'; %layer anterior à layer de classificaçao

trainingFeatures = activations(convnet, trainingset, 15,'MiniBatchSize', 300, 'OutputAs', 'columns');

%%
% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.

classifier = fitcecoc(trainingFeatures, trainingset.Labels,'Learners', 'Linear','Coding', 'onevsall', 'ObservationsIn', 'columns');

%%
% Extract test features using the CNN
testFeatures = activations(convnet, testset, 15, 'MiniBatchSize',64);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

%% Tabulate the results using a confusion matrix.
confMat = confusionmat(testset.Labels, predictedLabels)
