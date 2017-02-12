function Output= elm_reg_pred(test_features,InputWeight,BiasofHiddenNeurons,OutputWeight,ActivationFunction)

P=test_features';
tempH=InputWeight*P;
NumberofTrainingData=size(P,2);
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind); 
tempH=tempH+BiasMatrix;

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here                
end
clear tempH;        

Output=(H' * OutputWeight)';  