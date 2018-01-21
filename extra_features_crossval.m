%general features extraction

clear all;
close all;
%%
tipo = {'.png1','.png2','.png3','.png4','.png5','.png6','.png7'};

% armazenamento das imagens segundo uma estrutura imagaDatastore (aqui vao
% ser percorridas tanto a PastaBOCAS como a PastaFACES e a PastaOLHOS
% segundo as categorias .png1,.png2,etc
imds = imageDatastore('C:\Users\danis\Documents\Base de dados\projecto 2\testado\PastaBOCAS','FileExtensions',{'.png1','.png2','.png3','.png4','.png5','.png6','.png7'});

labels=zeros(327,1);
features_treino=[];

feat=cell(1,327);

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
    
    % Features extraction (aqui terá de ser executado cada extractor de
    % forma individual (colocando em comentario o codigo respeitante aos
    % restantes)

    % 1) deteçao de cantos: Features Harris
    ptsOriginal  = detectHarrisFeatures(image_bw);
    % (tamanho da celula que realizara extraçao da feature pode ser
    % modificada no argumento 'cellsize')
    [hog2, validPoints, ptVis] = extractHOGFeatures(image_bw, ptsOriginal,'CellSize',[2 2]); 
    
    %utilizaçao das 2 primeiras features encontradas para cada imagem
    hog2=reshape(hog2(1:2,:),1,36*2);
    features_treino =  [features_treino; hog2];
    
    % 2) features HOG (imagem completa)
    %             [hog_8x8, vis8x8]= extractHOGFeatures(image_gamma, 'CellSize', [8 8]); % extraçao de features é feita a partir de celulas de 8x8.
    %             features_treino =  [features_treino; hog_8x8];
    
    % 3) features SURF
    %         ptsOriginal  = detectSURFFeatures(image_bw);% mesma subtituiçao aplicada na imagem teste, para diferente metodo utilizado
    %         strongest = selectStrongest(ptsOriginal, 2);
    %         [feat_orig,validPtsOriginal] = extractFeatures(image_bw, strongest);
    %         feat_orig=reshape(feat_orig,1,size(feat_orig,1)*size(feat_orig,2));
    %         features_treino = [features_treino; feat_orig];
    
    % 4) features SIFT
    % dificuldades em detectar features em determinadas imagens,
    % impossibilitando a avaliaçao da sua performance
    %         feat_SIFT=SIFT_feature(image_bw);
    %         features_treino = [features_treino feat_SIFT(:,1)];
    
    % 5) Gabor wavelets
    %         gaborArray = gaborFilterBank(5,8,39,39); %Banco de 5*8 filtros com tamanho de 39*39 cada 
    %         featureVector = gaborFeatures(image,gaborArray,6,6); %vetor coluna com tamanho =(m*n*u*v)/(d1*d2)
    %         features_treino =  [features_treino; featureVector'];
    
end
%% -----------------------------------1o metodo de classificaçao-------------------------------------
% 1) SVM
% 2) K-NN
%% classificaçao SVM (com validaçao cruzada K-folds)

% dada a intenção do projecto, divide-se a base de dados num grupo de
% treino, a partir do qual se realizará uma validaçao cruzada K-Fold (divisao coordenada da base de dados em treino e teste) de forma
% a criar o modelo SVM para posterior classificaçao do grupo teste.
% Para além disto, dado o caracter da experiencia em si, onde a partir de
% uma base de dados uniforme (conjunto das emoçoes de interesse retiradas
% de cada individuo, para posterior classificaçao de novo individuo), se
% escolheram 11 individuos para o treino do modelo SVM, tendo sido
% realizada validaçao nos restantes 4.
%
% stat_values=[];
%
% TP_svm=zeros(1,7);
% TN_svm=zeros(1,7);
% FP_svm=zeros(1,7);
% FN_svm=zeros(1,7);
% for i=1:7
%     for k=1:327
%     %labeling das classes em estudo
%         newlabels(k,1)=isequal(labels(k,1),i);
%     end

%     %SVM classifier with cross validation. Kfold
%     %com 15 dados cada Fold de maneira a fazer uma divisao em termos de
%     %individuos pertencentes à base de dados
%     SVMModel = fitcsvm(features_treino,newlabels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto', 'KFold',15);
%
%     [svm_label,score] = kfoldPredict(SVMModel);
%
%     %AUC e curva ROC
%
%     figure('name','ROC curve')
%
%     [x,y,t,AUC_perf]=perfcurve(newlabels,score(:,2),1); %colocar grelha
%     plot(x,y,'r-');grid on;
%     legend('perfcurve');
%     title(['AUC= ',num2str(AUC_perf)]);
%
% determinaçao de sensibilidades e especificidades
%     TP=0;
%     TN=0;
%     FP=0;
%     FN=0;
%     for k=1:327
%         %se for positivo:
%         if (isequal(svm_label(k,1),1)==1)
%
%             %true positive
%             if(isequal(newlabels(k),1)==1)
%
%                 TP = TP + 1;
%                 %false positive
%             else
%
%                 FP=FP+1;
%             end
%
%             %negativo
%         else
%             if(isequal(newlabels(k),0)==1)
%
%                 TN = TN + 1;
%
%             else
%
%                 FN=FN+1;
%             end
%
%         end
%     end
%     TP_svm(i)=TP;
%     TN_svm(i)=TN;
%     FP_svm(i)=FP;
%     FN_svm(i)=FN;
%
%
%
%     specificity=TN_svm/(TN_svm+FP_svm);
%     sensitivity=TP_svm/(TP_svm+FN_svm);
%     accuracy=(TP_svm+TN_svm)/(TP_svm+FP_svm+FN_svm+TN_svm);
%     FP_rate=FP_svm/(TN_svm+FP_svm);
%
%     temp=[specificity,sensitivity,accuracy,FP_rate];
%     stat_values=cat(1,stat_values,temp);
%
% end
%%
% Sensitivity=stat_values(:,2);
% Specificity=stat_values(:,1);
% Accuracy=stat_values(:,3);
% EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
% EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
% %%
% table=table(Sensitivity, Specificity,Accuracy, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity', 'Specificity','Accuracy'})

%% classificador K-NN loo (k=1)

k=1;
[pred,percstats]=k_nn_loo(New_matrix',k,labels);

%matriz confusao a partir dos labels pred obtidos no knn e das labels reais
stats1 = confusionmatStats(labels,pred)

tab=round(percstats*100)

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
k1 = array2table(tab, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat1 = table(stats1.sensitivity,stats1.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

%% classificador K-NN loo (k=3)
k=3;
[pred,percstats]=k_nn_loo(New_matrix',k,labels);

stats3 = confusionmatStats(labels,pred)
tab=round(percstats*100);

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1','V2','V3','V4','V5','V6','V7'};
k3 = array2table(tab, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat3 = table(stats3.sensitivity,stats3.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

%% classificador K-NN loo (k=5)

New_matrix=features_treino';
k=5;
[pred,percstats]=k_nn_loo(New_matrix',k,labels);
stats5 = confusionmatStats(labels,pred)
tab=round(percstats*100);

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1','V2','V3','V4','V5','V6','V7'};
k5 = array2table(tab, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat5 = table(stats5.sensitivity,stats5.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

%% classificador K-NN loo (k=7)
k=7;
[pred,percstats]=k_nn_loo(New_matrix',k,labels);

stats7 = confusionmatStats(labels,pred)
tab=round(percstats*100);

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1','V2','V3','V4','V5','V6','V7'};
k7 = array2table(tab, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stat7 = table(stats7.sensitivity,stats7.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

%
% %% Visualization of the points distribution
% figure('Name','Points distribution according to Face features')
% plot3(New_matrix(1,1:11:43),New_matrix(2,1:11:43),New_matrix(3,1:11:43),'go',New_matrix(1,2:11:43),New_matrix(2,2:11:43),New_matrix(3,2:11:43),'y*',New_matrix(1,3:11:43),New_matrix(2,3:11:43),New_matrix(3,3:11:43),'ro',New_matrix(1,4:11:43),New_matrix(2,4:11:43),New_matrix(3,4:11:43),'b*')

%% Organizaçao dos resultados em 3 vetores coluna para 2o metodo de classificaçao
escolhidoBoca=pred;
%%
escolhidoOlhos=pred;
%%
escolhidoFace=pred;

%% -----------------------------------2o metodo de classificaçao-------------------------------------

resultadofinal = class_final(escolhidoBoca,escolhidoOlhos,escolhidoFace, 1, 1, 1);


statstotal = confusionmatStats(labels,resultadofinal)
tabtotal=round(statstotal.confusionMat);

EmotionsPred={'P1','P2','P3','P4','P5','P6','P7'};
EmotionsVer={'V1';'V2';'V3';'V4';'V5';'V6';'V7'};
ktotal= array2table(tabtotal, 'RowNames', EmotionsVer,'VariableNames',EmotionsPred)
stattotal = table(statstotal.sensitivity,statstotal.specificity, 'RowNames', EmotionsVer,'VariableNames',{'Sensitivity','Specificity'})

