function resultadofinal = class_final(vet1, vet2, vet3, peso1, peso2, peso3)

resultadofinal=zeros(length(vet1),1);

pesotot=peso1+peso2+peso3;

tot = zeros(1,pesotot);

for i=1:length(vet1)
    
    for p=1:peso1
        
        tot(1,p) = vet1(i);
        
    end
    
    for p=1:peso2
        
        tot(1,peso1+p) = vet2(i);
        
    end
    
    for p=1:peso3
        
        tot(1,peso2+peso1+p) = vet3(i);
        
    end
    
    
    
    resultadofinal(i,1) = mode(tot);
    
end

end