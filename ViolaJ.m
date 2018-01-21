function N = ViolaJ()
clear all;
close all;

%numero de imagens
N = 327;

% Queremos 3 matrizes: cara, olhos e boca
%caras -> 64x64 = 4096 pixeis
%boca e olhos -> 25x50 = 1250 p
% matcaras = zeros(N,4096);
% matolhos = zeros(N,1250);
% matbocas = zeros(N,1250);

%labels - i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
% labels = zeros(327,1);

%cria as pastas para guardar as imagens:
mkdir('PastaFACES');
mkdir('PastaBOCAS');
mkdir('PastaOLHOS');

%escolha da directoria onde vai guardar as imagens
cd('C:\Users\danis\Desktop')  ;

%vetor com diretorias das pastas nesta diretoria
P1 = subdir('C:\Users\danis\Desktop\Emotion');


l=0;


for i=1:length(P1)
    
    d = P1{1,i};
    P2 = subdir(d);
    
    for j=1:length(P2)
        
        f = P2{1,j};
        F = dir(f);
        
        if length(F)==3
            
            l=l+1;
            cd(f);
            nome= F(3, 1).name;
            fid = fopen(nome);
            A = textscan(fid,'%f');
            labels(l) = A{1,1};
            fclose('all');
            
            %vai buscar a imagem correspondente
            fnovo = strsplit(f,'Emotion');
            f4 = strcat(fnovo(1),'cohn-kanade-images',fnovo(2));
            
            %diretoria da img ->  f4{1} e nome -> nomeimg
            cd(f4{1})  ;
            nomeimg = strsplit(nome,'_emotion.txt');
            nomeimg = strcat(nomeimg(1),'.png');
            
            imagem = imread(nomeimg{1});
            %faz o viola jones e adiciona a matriz
            
            %cara(comando viola jones para deteçao de cara)
            FDetect = vision.CascadeObjectDetector;
            
            
            BBF = step(FDetect,imagem);
            
            % clausula responsavel pela precauçao no caso de nenhum
            % elemento facial ser capturado
            if length(BBF)==0
                BBF = BBF_velho;
            end
            
            for k = 1:size(BBF,1)
                F = imcrop(imagem,BBF(k,:));
            end
            
            
            %boca (comando viola jones para deteçao de boca)
            MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',60);
            MouthDetect.UseROI = true;
            [y1,x1,z1] = size(F);
            ROI1 = [round(x1*0.2),round(y1*0.63),round(x1*0.6),round(y1*0.37)];

            BBM=step(MouthDetect,F,ROI1);
            
            % clausula responsavel pela precauçao no caso de nenhuma
            % boca ser capturada
            if length(BBM)==0
                BBM = BBM_velho;
            end
            for k = 1:size(BBM,1)
                M = imcrop(F,BBM(k,:));
            end
            
              
            %olhos (comando viola jones para deteçao de olhos)
            EyeDetect = vision.CascadeObjectDetector('EyePairBig');
            EyeDetect.UseROI = true;
            ROI2 = [round(x1*0.1),round(y1*0.15),round(x1*0.8),round(y1*0.45)];
            BBE=step(EyeDetect,F,ROI2);

            pause(5)
            
            % clausula responsavel pela precauçao no caso de nenhuns
            % olhos serem capturados
            if length(BBE)==0
                BBE = BBE_velho;
            end
            
            for k = 1:size(BBE,1)
                E = imcrop(F,BBE(k,:));
            end
            
            %tamanho final da fac,boca e olhos de 64x64,25x50,25x50
            F = imresize(F,[64,64]);
            M = imresize(M,[25,50]);
            E = imresize(E,[25,50]);
            
               figure(1)
           subplot(1,3,1);
           imshow(F);
           subplot(1,3,2);
           imshow(E);
           subplot(1,3,3);
           imshow(M);
           
           % armazenamento das imagens na nova directoria 
            cd('C:\Users\danis\Desktop\PastaFACES');
            nomecomlabel = strcat(nomeimg{1},num2str(labels(l)));
            imwrite(F,nomeimg{1});
            movefile(nomeimg{1},nomecomlabel);
            cd('C:\Users\danis\Desktop\PastaBOCAS');
            nomecomlabel = strcat(nomeimg{1},num2str(labels(l)));
            imwrite(M,nomeimg{1});
            movefile(nomeimg{1},nomecomlabel);
            cd('C:\Users\danis\Desktop\PastaOLHOS');
            nomecomlabel = strcat(nomeimg{1},num2str(labels(l)));
            imwrite(E,nomeimg{1});
            movefile(nomeimg{1},nomecomlabel);
  
            BBF_velho = BBF;
            BBM_velho = BBM;
            BBE_velho = BBE;
            
       
            
        end
        
        %volta a diretoria original
        cd('C:\Users\danis\Desktop')  ;
    end
    
end
end

