function [k_nn,finaltotal]=k_nn_loo(test,k,labels)
% test=features_treino;
% k=5;
k_nn=zeros(size(test,1),1);

for test_r=1:size(test,1)
    dis=zeros(size(test,1),2);
    tot_min=zeros(k,2);
    for train_r=1:size(test,1)
        if (isequal(test_r,train_r)==1)
            continue;
        else
            dis(train_r,:)=[sqrt(sum((test(test_r,:)-test(train_r,:)).^2)) labels(train_r)];
            
        end
    end
    
    sort_asc = sortrows(dis,1); %pre defined as 'ascend'
    for n=1:k
        tot_min(n,:)=[sort_asc(n+1,1) sort_asc(n+1,2)];
        
    end
    %labels each test sample
    k_nn(test_r,1)=mode(tot_min(:,2));
      
end

C = confusionmat(labels,k_nn);
for i=1:7
    for j=1:7
        finaltotal(i,j)=C(i,j)/sum(C(i,:));
    end
end
%%
% test=feat;
% % % k=5;
% k_nn=zeros(size(test,1),1);
% % %test=cell(1,x);
% for test_r=1:size(test,2)
%     test=reshape(test{1,test_r},1,size(test{1,test_r},1)*size(test{1,test_r},2));
%     dis=zeros(size(test,2),2);
%     tot_min=zeros(k,2);
%     for train_r=1:size(test,1)
%         if (any(test(test_r,:)==test(train_r,:))==1)
%             continue;
%         else
%             dis(train_r,:)=[sqrt(sum((test(test_r,:)-test(train_r,:)).^2)) labels(train_r)];
%             
%         end
%     end
%     
%     sort_asc = sortrows(dis,1); %pre defined as 'ascend'
%     for n=1:k
%         tot_min(n,:)=[sort_asc(n+1,1) sort_asc(n+1,2)];
%     end
%     
%     %labels each test sample
%     k_nn(test_r,1)=mode(tot_min(:,2));
%       
% end
% 
% C = confusionmat(labels,k_nn);
% for i=1:7
%     for j=1:7
%         finaltotal(i,j)=C(i,j)/sum(C(i,:));
%     end
% end