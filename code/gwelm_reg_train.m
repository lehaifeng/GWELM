function [InputWeight,BiasofHiddenNeurons,OutputWeight] = gwelm_reg_train(train_data,NumberofHiddenNeurons, ActivationFunction,w)
% train_data=all_train_data(:,1:5);
% NumberofHiddenNeurons=30;
% ActivationFunction='sig';
% w=weight;

T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;         
NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P; 
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

for i=1:length(w);
    H_weight(:,i)=w(i)^0.5*H(:,i);
    T_weight(1,i)=w(i)^0.5*T(1,i);
end  
clear H;clear T;
H=H_weight;
T=T_weight;

 OutputWeight=pinv(H') * T';
 

