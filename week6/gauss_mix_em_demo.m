function [Ptrain,Ptest] = gauss_mix_em_demo(X,Xtest,M,Max_Its)

[N,D]=size(X);

%Randomly initiliase the Posteriors in the E-step
P=rand(N,M);
P=P./repmat(sum(P')',1,M);

%Store the likelihood of the data under the mixture model in this matrix
%for plotting during the EM run
LL = [];

%This is the main EM algorithm

for n=1:Max_Its
    for m=1:M
        %M_step
        MU_hat(m,:)= sum(X.*repmat(P(:,m),1,D))./sum(P(:,m));
        XT = (X - repmat(MU_hat(m,:),N,1)).*repmat(P(:,m),1,D);
        Cm = (XT'*(X - repmat(MU_hat(m,:),N,1)))./sum(P(:,m));
        C_hat(m,:) = reshape(Cm,D^2,1)'; 
        Pr(m) = mean(P(:,m));
        %E_step
        P(:,m) = gauss(MU_hat(m,:),reshape(C_hat(m,:),D,D),X).*Pr(m);
    end
    
    %compute the likelihood
    Ptrain = mean(log(sum(P')));
    LL=[LL;Ptrain];
    
    %Normlise the Posterior
    P=P./repmat(sum(P')',1,M);
    
    %Nice Demo Graphics
    if D==2
        [x1,x2]=meshgrid(-6:0.1:6,-6:0.1:6);
        [n,n]=size(x1);
        XX=[reshape(x1,n*n,1) reshape(x2,n*n,1)];
        subplot(121)
        
        %compute the log likelihood
        pt=0;
        for m=1:M
            pt=pt + Pr(m)*gauss(MU_hat(m,:)',reshape(C_hat(m,:),D,D),XX);
        end
        Ptest = mean(log(pt));
        %plot the isocontours of likelihood
        contour(x1,x2,reshape(pt,[n,n]));
        hold on;
        plot(X(:,1),X(:,2),'.'); drawnow
        hold off;
        pause(0.1)  
        subplot(122)
        plot(LL);drawnow
        title('Data Log Likelihood')
    end
    %Compute Test Data Likelihood
    pt=0;
    for m=1:M
       pt=pt + Pr(m)*gauss(MU_hat(m,:)',reshape(C_hat(m,:),D,D),Xtest);
    end
    Ptest = mean(log(pt));
end