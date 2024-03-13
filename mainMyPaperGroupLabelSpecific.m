clear;clc;close all;

addpath(genpath('./'))
str = 'stackexcooking';
load(str);

Y = train_target;
Y(train_target<0) = 0;

ii = 1;
%lambda1->[0.01,0.1,1,10,100] and lambda2->[0.01,0.1,1,10,100], parameters are not fixed
lambda1 = 0.01;
lambda2 = 0.01;

[~,W] = PNCMLFS(train_data,Y',lambda1,lambda2);

d = size(train_data, 2);
c = size(Y, 1);

for groupNum = [ceil(c/2),ceil(c/4),ceil(c/6)]
    [idx,cc]=kmeans(Y, groupNum,'Start',Y(1:groupNum,:), 'MaxIter',1000);
    for k = [5,10,15,20,25,30,35,40,45,50,55,60]
        HammingLoss(ii,2) = k;
        RankingLoss(ii,2) = k;
        OneError(ii,2) = k;
        Coverage(ii,2) = k;
        Average_Precision(ii,2) = k;

        HammingLoss(ii,1) = groupNum;
        RankingLoss(ii,1) = groupNum;
        OneError(ii,1) = groupNum;
        Coverage(ii,1) = groupNum;
        Average_Precision(ii,1) = groupNum;
        
        if k>d
            break;
        end

        newW = W;
        for i = 1:c
            [~, order] = sort(W(:,i),'descend');
            for j = k+1:d
                newW(order(j),i) = 0;
            end
        end

        newW2 = W;
        for i=1:groupNum
            [~, order] = sort(sum(W(:,idx == i),2),'descend');
            for j = k+1:d
                newW2(order(j),idx == i) = 0;
            end
        end

        [~, order] = sort(sum(newW+newW2,2),'descend');

        numFeature = size(train_data,2);
        if numFeature>1000
            numSeleted = round(numFeature * 0.1);
        elseif numFeature<=1000 && numFeature>500
            numSeleted = round(numFeature * 0.2);
        elseif numFeature<=500 && numFeature>100
            numSeleted = round(numFeature * 0.3);
        else
            numSeleted = round(numFeature * 0.4);
        end

        selFeature = order(1:numSeleted);
        Num=10;Smooth=1;
        [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,selFeature),train_target,Num,Smooth);
        [HammingLoss(ii,3),RankingLoss(ii,3),OneError(ii,3),Coverage(ii,3),Average_Precision(ii,3),Outputs,Pre_Labels]=MLKNN_test(train_data(:,selFeature),train_target,test_data(:,selFeature),test_target,Num,Prior,PriorN,Cond,CondN);
        ii=ii+1;
    end
end

filename = ['PNCMLFSLS_NEWAllParameter_Group ' str];
save(filename, 'HammingLoss', 'RankingLoss','OneError','Coverage','Average_Precision','lambda1','lambda2');