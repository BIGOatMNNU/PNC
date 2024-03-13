
clear;clc;close all;

% 'Arts_data','Business_data','Computer_data','Education_data','Health_data','Recreation_data','Reference_data','Science_data','Socail_data','Society_data','Birds_data','CAL500_data','Emotions_data','Enron_data','Image_data','Langlog_data','Medical_data','Scene_data','Slashdot_data','Yeast_data','Corel5k_data','Bibtex_data'

addpath(genpath('./'))
str = {'sample'};
for ii = 1:length(str)
    load(str{ii});
    Y = train_target;
    Y(train_target<0) = 0;
    i = 1;
    for lambda1 = [0.01,0.1,1,10,100]
        for lambda2 = [0,0.01,0.1,1,10,100]
            
            
            HammingLoss(i,1) = lambda1;
            HammingLoss(i,2) = lambda2;
            RankingLoss(i,1) = lambda1;
            RankingLoss(i,2) = lambda2;
            OneError(i,1) = lambda1;
            OneError(i,2) = lambda2;
            Coverage(i,1) = lambda1;
            Coverage(i,2) = lambda2;
            Average_Precision(i,1) = lambda1;
            Average_Precision(i,2) = lambda2;

            feature_slct = PNCMLFS(train_data,Y',lambda1,lambda2);
%             feature_slct = thirdPaperMethod2(train_data,Y');

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
            
            selFeature = feature_slct(1:numSeleted);
            Num=10;Smooth=1;
            [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,selFeature),train_target,Num,Smooth);
            [HammingLoss(i,3),RankingLoss(i,3),OneError(i,3),Coverage(i,3),Average_Precision(i,3),Outputs,Pre_Labels]=MLKNN_test(train_data(:,selFeature),train_target,test_data(:,selFeature),test_target,Num,Prior,PriorN,Cond,CondN);
            i=i+1;
        end
    end
    
    filename = ['PNCMLFS_NEWAllParameter ' str{ii}];
    save(filename, 'HammingLoss', 'RankingLoss','OneError','Coverage','Average_Precision');
    clear HammingLoss;
    clear RankingLoss;
    clear OneError;
    clear Coverage;
    clear Average_Precision;
end