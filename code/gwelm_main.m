function [fitted_results,predicting_results,CVscore,rmse_fit,rmse_pre]=gwelm_main(STdata,location,hidden_num, pre_length,step_length)
% Usage: [fitted_results,predicting_results,p_fit,p_gw]=GWELM_main(ST_Data_File, Location_Data_File, hidden_num, activationFunction,pre_length)
%
% Input:
% STdata                      - Spatio-temporal series data Cell{1:L}£¨1:M,1:N£©
%                               L: number of stations;
%                               M: length of time series; N: number of variate 
%                              (1st column: predictive variable and 2:L column:covariate)
% location                    - Filename of location information 
%                               Matrix(2*N) the coordinates: X and Y
% hidden_num                  - Number of hidden neurons assigned to the GWELM
% pre_length                  - Length of time series of validate data
% Step_length                 - (Step_length) step ahead predictor


% Output:
% fitted_results              - Fitted results of training data
% predicting_results          - Predicting results of validate data
% CVscore                     - Value of CV function
% The code is edited based on the code of ELM toolbox (http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm)

L=length(STdata);
[M,N]=size(STdata{1});

%%%%%%%%%%% Compute spatial distance  (Euclidean distance)
for i=1:L
    for j=1:L
        dis(i,j)=sqrt((location(1,i)-location(1,j))^2+(location(2,i)-location(2,j))^2);
    end
end

%%%%%%%%%%% Apply GWELM to each station
hwait=waitbar(0,'Please wait>>>>>>>>');
for station=1:L
    dis_vector=dis(station,:);
    min_dis_station=min(dis_vector(find(dis_vector~=0)));
    for k=1:10
        bandwidth=2*k*min_dis_station;
        [candida_fitted_results(station,k,1:M-pre_length),candida_predicting_results(station,k,1:pre_length),CVscore(station,k)]=gwelm_reg(STdata,station,hidden_num,pre_length,dis,bandwidth,step_length);
    end
        str=['Running now ! ',num2str(floor(station*100/L)),'%'];
        waitbar(station/L,hwait,str);
        pause(0.005);
end
close(hwait);

%%%%%%%%%%% Select the optimal bandwidth 
for i=1:L
temp=CVscore(i,:);
k=find(temp==min(temp));
fitted_results(i,1:M-pre_length)=candida_fitted_results(i,k,:);
predicting_results(i,1:pre_length)=candida_predicting_results(i,k,:);
end


%%%%%%%%%%% Compute the error 
for i=1:L
    temp=STdata{i}(:,1);
    observation(i,1:M)=temp;
end
for i=1:M-pre_length
    [mae_fit(i),rmse_fit(i),rse_fit(i),nmse_fit(i)]=accuracy(observation(:,i)',fitted_results(:,i)');
end
for i=1:pre_length
    [mae_pre(i),rmse_pre(i),rse_pre(i),nmse_pre(i)]=accuracy(observation(:,M-pre_length+i)',predicting_results(:,i)');
end






