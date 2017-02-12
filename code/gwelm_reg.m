function [fittting_value,output_gw,pred_train,p_gw]=gwelm_reg(Data, station,num_hidden,pre_day,Dis,bandwidth,step)

%%%%%%%%%%% Partition data
Time_split=pre_day;
for i=1:length(Data)
    [m,n]=size(Data{i});
    beg=(i-1)*(m-Time_split)+1;
    over=i*(m-Time_split);
    temp_data=Data{i};
    train_data(beg:over,1:n)=temp_data(1:m-Time_split,1:n);
    test_data(i,1:Time_split,1:n)=temp_data(m-Time_split+1:m,1:n);
end
clear temp_data;
clear beg;
clear over;
clear i;
%%%%%%%%%%% Delete NaN
index_nan=any(isnan(train_data),2);
train_data(any(isnan(train_data),2),:)=[];

%%%%%%%%%%% Data Normaliztion
[m1,n1]=size(train_data);
for j=1:n1
    min_final(j)=min(train_data(:,j));
    max_final(j)=max(train_data(:,j));
end
len1=length(train_data);
for i=1:n1
    train_data_norm(1:len1,i)=(train_data(:,i)-min_final(i))/(max_final(i)-min_final(i))*2-1;
end
clear all_train_data;
train_data=train_data_norm;

%%%%%%%%%%% define weight matrix
temp_weight=[];
l=m-Time_split;
for i=1:length(Data)
    if Dis(i,station)<=bandwidth
        temp_w=(1-(Dis(i,station)/bandwidth)^2)^2;
    else
        temp_w=0;
    end
    % temp_w=exp(-(Dis(i,station)/bandwidth)^2);
    num_station=l-sum(index_nan((i-1)*l+1:i*l));
    vector_w=temp_w*(zeros(1,num_station)+1);
    temp_weight=[temp_weight,vector_w];
end
weight=temp_weight;

%%%%%%%%%%% train GWELM
[InputWeight,BiasofHiddenNeurons,OutputWeight] =gwelm_reg_train(train_data(:,1:n), num_hidden, 'sig',weight);
% activationFunction          - Type of activation function:
%                                'sig' for Sigmoidal function
%                                'sin' for Sine function
%                                'hardlim' for Hardlim function

%%%%%%%%%%% Compute fitting precision
real_station=Data{station}(1:m-Time_split,1:n);
real_station(any(isnan(real_station),2),:)=[];

for i=1:n1
    real_station_norm(:,i)=(real_station(:,i)-min_final(i))/(max_final(i)-min_final(i))*2-1;
end
fittting_value_norm=elm_reg_pred(real_station_norm(:,2:n),InputWeight,BiasofHiddenNeurons,OutputWeight,'sig');
fittting_value=(fittting_value_norm+1)*0.5*(max_final(1)-min_final(1))+min_final(1);
pred_train=sum(abs(fittting_value(:,1)-real_station(:,1)));

%%%%%%%%%%% all fitting precision
all_fittting_value= elm_reg_pred(train_data(:,2:n),InputWeight,BiasofHiddenNeurons,OutputWeight,'sig');
all_values=(all_fittting_value+1)*0.5*(max_final(1)-min_final(1))+min_final(1);
real_values=(train_data(:,1)+1)*0.5*(max_final(1)-min_final(1))+min_final(1);
pred_train=sum(abs(real_values(:,1)-all_values(:,1)))-sum(abs(fittting_value(:,1)-real_station(:,1)));


%%%%%%%%%%% Predict
feature(1:Time_split,1:n-1)=test_data(station,1:Time_split,2:n);

%%%%%%%%%%% Repair NaN
for i=1:Time_split
    for j=1:n-1
        k=i;
        if isnan(feature(i,j))==1 && k<=Time_split/2
            while isnan(feature(k+1,j))==1
                k=k+1;
            end
            feature_norm(i,j)=(feature(k+1,j)-min_final(j+1))/(max_final(j+1)-min_final(j+1))*2-1;
        elseif isnan(feature(i,j))==1 && k>Time_split/2
            while isnan(feature(k-1,j))==1
                k=k-1;
            end
            feature_norm(i,j)=(feature(k-1,j)-min_final(j+1))/(max_final(j+1)-min_final(j+1))*2-1;
        else
            feature_norm(i,j)=(feature(k,j)-min_final(j+1))/(max_final(j+1)-min_final(j+1))*2-1;
        end
    end
end
clear feature
feature=feature_norm;

%%%%%%%%%%%  predict the validate data
output(1)= elm_reg_pred(feature(1,1:n-1),InputWeight,BiasofHiddenNeurons,OutputWeight,'sig');
for i=2:Time_split
    output(i)= elm_reg_pred(feature(i,1:n-1),InputWeight,BiasofHiddenNeurons,OutputWeight,'sig');
end

%%%%%%%%%%% Inverse normaliztion
output_real=(output+1)*0.5*(max_final(1)-min_final(1))+min_final(1);
clear output
output_gw=output_real;

%%%%%%%%%%% precision evaluation
for i=1:step
    real_y(i)=sum(test_data(station,i:step:Time_split,1));
    pred_y(i)=sum(abs(output_gw(i:step:Time_split)-test_data(station,i:step:Time_split,1)));
    p_gw(i)=1-pred_y(i)/real_y(i);
end

end