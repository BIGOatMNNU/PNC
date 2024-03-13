function [order, W] = PNCMLFS(X,Y,lambda1,lambda2)
%   

Y(Y==-1)=0;

m = size(X, 1);
d = size(X, 2);
c = size(Y, 2);

eps = 1e-12;
% WAll = zeros(d,c);
L0 = 1 - pdist2( Y', Y', 'cosine' );
ind=find(isnan(L0));
L0(ind)=0;
LAll = L0 - diag(diag(L0));
LAll_hat = diag(sum(LAll,2));
LAll = LAll_hat - LAll;


k=2;
numB=k; 
eps = 1e-12;
%positive or negative correlation
for i=1:d
    Xtemp = X(:,i);
    u = unique(Xtemp);
    if (length(u) == 1) && (u == 1 || u==0) %discrete value
        numB = 2;
        Xtemp = Xtemp + 1;
    elseif (length(u) == 2) && (~isempty(find(u==1, 1)) && ~isempty(find(u==0, 1)))%discrete value
        numB = 2;
        Xtemp = Xtemp + 1;
    else
        numB=k; 
        [Xtemp, ~]=trans(Xtemp,[],numB);
    end
    for j=1:c
		% positive sample
        indPos = find(Y(:,j) == 1);
        for k=1:numB
            pos(k)=length(find(Xtemp(indPos)==k))/length(indPos);
        end
        %negative sample
        indNeg = find(Y(:,j) ~= 1);
        for k=1:numB
        	neg(k)=length(find(Xtemp(indNeg)==k))/length(indNeg);
        end
        
        %positive or negative entropy
        posEntropy = 0;
         posCorr = 0;
        if ~isempty(indPos)
            for k=1:numB
                if numB==2
                    posEntropy= posEntropy-pos(k)*log2(max(pos(k),eps));
                else
                    posEntropy= posEntropy-pos(k)*log(max(pos(k),eps))/log(numB);
                end
            end
            posCorr  = 1-posEntropy;
        end
        negEntropy = 0;
         negCorr = 0;
        if ~isempty(indNeg)
            for k=1:numB
                if numB==2
                    negEntropy= negEntropy-neg(k)*log2(max(neg(k),eps));
                else
                    negEntropy= negEntropy-neg(k)*log(max(neg(k),eps)/log(numB));
                end
            end
            negCorr  = 1-negEntropy;
        end
        
        FL(i,j) = posCorr-negCorr;
%         if posEntropy>=negEntropy
%             FL(i,j) = posEntropy;
%         else
%             FL(i,j) = -negEntropy;
%         end
    end 
end

iter = 1;
W = ones(d,c)*.5;
H = ones(c,c)*.5;
% Wtest = (X'*X)\(X'*Y);
% FL(Wtest<0) = -FL(Wtest<0);

while(1)
    %update F

    F = (Y+X*W)/(2*eye(c)+lambda1*LAll);
    
    
    %update W
    HW = diag(0.5./max(sqrt(sum((W).*(W),2)),eps));
    W = (X'*X+lambda2*eye(d)+HW)\(X'*F+lambda2*FL*H);
%     A = X'*X+HW;
%     B = lambda2*L0*L0';
%     C = -lambda2*FL*L0'-X'*F;
%     W = lyap(A,B,C);
    
    %update H
    H = pinv(FL'*FL)*(FL'*W);

    obj(iter) = (norm((X*W - F), 'fro'))^2+sum(sqrt(sum((W).*(W),2)))...
        +lambda2*(norm((W - FL*H), 'fro'))^2+...
        (norm((Y - F), 'fro'))^2+lambda1*trace(F*LAll*F');
    disp(iter+":"+obj(iter));
    if (iter>=2 && abs(obj(iter-1)-obj(iter))<=1e-2) || iter>1000
        break;
    end
    iter = iter+1;
end

W = abs(W);


[~, order] = sort(sum(W,2),'descend');
end


