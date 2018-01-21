
clear all;
close all;
%%
tipo = {'.png1','.png2','.png3','.png4','.png5','.png6','.png7'};


% armazenamento das imagens segundo uma estrutura imagaDatastore (aqui vao
% ser percorridas tanto a PastaBOCAS como a PastaFACES e a PastaOLHOS
% segundo as categorias .png1,.png2,etc

% imds = imageDatastore('C:\Users\danis\Documents\Base de dados\projecto 2\testado\PastaOLHOS','FileExtensions',{'.png1','.png2','.png3','.png4','.png5','.png6','.png7'});
% imds = imageDatastore('C:\Users\danis\Documents\Base de dados\projecto 2\testado\PastaBOCAS','FileExtensions',{'.png1','.png2','.png3','.png4','.png5','.png6','.png7'});
imds = imageDatastore('C:\Users\danis\Documents\Base de dados\projecto 2\testado\PastaFACES','FileExtensions',{'.png1','.png2','.png3','.png4','.png5','.png6','.png7'});

labels=zeros(327,1);
features_treino=[];

% matolhos=[];
%     matbocas=[];
    matcaras=[];

for i=1:327
    
    %label da imagem
    label=str2double(imds.Files{i,1}(end));
    labels(i)=label;
    nome=imds.Files{i,1};
    
    %PRE PROCESSAMENTO DA IMAGEM (normalizaçao da imagem para gray-scale seguida de
    %aplicaçao da correçao gamma)
    hgamma = vision.GammaCorrector(2.0,'Correction','De-gamma');
    image = imread(nome);
    image=image(:,:,1);
    image_bw=mat2gray(image);
    image_gamma = step(hgamma, image_bw);
    
    image_gamma=reshape(image_gamma, [1,size(image_gamma,1)*size(image_gamma,2)]);
%     matolhos=[matolhos; image_gamma];
%         matbocas=[matbocas; image_gamma];
        matcaras=[matcaras; image_gamma];
end

%%
xF = double(matcaras);
xE = double(matolhos);
xM = double(matbocas);

%subtract meanif i==1

imeanF=mean(xF,1);
imeanE=mean(xE,1);
imeanM=mean(xM,1);

xF=bsxfun(@minus, xF, imeanF);
xE=bsxfun(@minus, xE, imeanE);
xM=bsxfun(@minus, xM, imeanM);

% calculo covariance
sF = cov(xF);
sE = cov(xE);
sM = cov(xM);

% obter eigenvalue & eigenvector
[VF,DF] = eig(sF);
eigvalF = diag(DF);

[VE,DE] = eig(sE);
eigvalE = diag(DE);

[VM,DM] = eig(sM);
eigvalM = diag(DM);

% sortear eigenvalues em ordem descendente
eigvalF = eigvalF(end:-1:1);
VF = fliplr(VF);

eigvalE = eigvalE(end:-1:1);
VE = fliplr(VE);

eigvalM = eigvalM(end:-1:1);
VM = fliplr(VM);
%% visualizaçao das imagens apos reshape
% show 0th through 15th principal eigenvectors
eig0F = reshape(imeanF, [64,64]);
figure,subplot(4,4,1)
imagesc(eig0F)
colormap gray
for i = 1:7
    subplot(4,4,i+1)
    imagesc(reshape(VF(:,i),64,64))

end

eig0E = reshape(imeanE, [50,25]);
figure,subplot(4,4,1)
imagesc(eig0E)
colormap gray
for i = 1:7
    subplot(4,4,i+1)
    imagesc(reshape(VE(:,i),50,25))
end


eig0M = reshape(imeanM, [50,25]);
figure,subplot(4,4,1)
imagesc(eig0M)
colormap gray
for i = 1:7
    subplot(4,4,i+1)
    imagesc(reshape(VM(:,i),50,25))
end

%% Consideram-se primeiros 7 valores proprios, que no caso ideal correspondem aos referenciais que definem as 7 emoçoes em estudo

New_matrixF=VF(:,1:7)'*double(matcaras)';
New_matrixE=VE(:,1:7)'*double(matolhos)';
New_matrixM=VM(:,1:7)'*double(matbocas)';

%% -----------------------------------1o metodo de classificaçao-------------------------------------

%% classificador K-NN loo (k=1)

k=1;
[knn1F,MCknn1F]=k_nn_loo(New_matrixF',k,labels);
[knn1E,MCknn1E]=k_nn_loo(New_matrixE',k,labels);
[knn1M,MCknn1M]=k_nn_loo(New_matrixM',k,labels);

stats1F = confusionmatStats(labels,knn1F)
tab1F=round(MCknn1F*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k1F = array2table(tab1F, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat1F = table(stats1F.sensitivity,stats1F.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats1E = confusionmatStats(labels,knn1E)
tab1E=round(MCknn1E*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k1E = array2table(tab1E, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat1E = table(stats1E.sensitivity,stats1E.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats1M = confusionmatStats(labels,knn1M)
tab1M=round(MCknn1M*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k1M = array2table(tab1M, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat1M = table(stats1M.sensitivity,stats1M.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})
%% classificador K-NN loo (k=3)

k=3;
[knn3F,MCknn3F]=k_nn_loo(New_matrixF',k,labels);
[knn3E,MCknn3E]=k_nn_loo(New_matrixE',k,labels);
[knn3M,MCknn3M]=k_nn_loo(New_matrixM',k,labels);

stats3F = confusionmatStats(labels,knn3F)
tab3F=round(MCknn3F*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k3F = array2table(tab3F, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat3F = table(stats3F.sensitivity,stats3F.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats3E = confusionmatStats(labels,knn3E)
tab3E=round(MCknn3E*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k3E = array2table(tab3E, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat3E = table(stats3E.sensitivity,stats3E.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats3M = confusionmatStats(labels,knn3M)
tab3M=round(MCknn3M*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k3M = array2table(tab3M, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat3M = table(stats3M.sensitivity,stats3M.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})
%% classificador K-NN loo (k=5)

k=5;
[knn5F,MCknn5F]=k_nn_loo(New_matrixF',k,labels);
[knn5E,MCknn5E]=k_nn_loo(New_matrixE',k,labels);
[knn5M,MCknn5M]=k_nn_loo(New_matrixM',k,labels);

stats5F = confusionmatStats(labels,knn5F)
tab5F=round(MCknn5F*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k5F = array2table(tab5F, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat5F = table(stats5F.sensitivity,stats5F.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats5E = confusionmatStats(labels,knn5E)
tab5E=round(MCknn5E*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k5E = array2table(tab5E, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat5E = table(stats5E.sensitivity,stats5E.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats5M = confusionmatStats(labels,knn5M)
tab5M=round(MCknn5M*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k5M = array2table(tab5M, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat5M = table(stats5M.sensitivity,stats5M.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})
%% classificador K-NN loo (k=7)

k=7;
[knn7F,MCknn7F]=k_nn_loo(New_matrixF',k,labels);
[knn7E,MCknn7E]=k_nn_loo(New_matrixE',k,labels);
[knn7M,MCknn7M]=k_nn_loo(New_matrixM',k,labels);

stats7F = confusionmatStats(labels,knn7F)
tab7F=round(MCknn7F*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k7F = array2table(tab, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat7F = table(stats.sensitivity,stats.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats7E = confusionmatStats(labels,knn7E)
tab7E=round(MCknn7E*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k7E = array2table(tab, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat7E = table(stats.sensitivity,stats.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

stats7M = confusionmatStats(labels,knn7M)
tab7M=round(MCknn7M*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k7M = array2table(tab, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat7M = table(stats.sensitivity,stats.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

%% -----------------------------------2o metodo de classificaçao-------------------------------------

resultadofinal = class_final(knn5F,knn5E,knn5M, 1, 1, 10)


statstotal = confusionmatStats(labels,resultadofinal)
tabtotal=round(statstotal.confusionMat)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
ktotal= array2table(tabtotal, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stattotal = table(statstotal.sensitivity,statstotal.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})
