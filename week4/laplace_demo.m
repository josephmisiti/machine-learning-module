clear
%A small constant used to avoid log of zero problems
SMALL_NOS = 1e-200;

%Define Sample size and data dimensionality
N=30;
D=2;

%Limits and grid size for contour plotting
Range=8;
Step=0.1;

%Two classes will have 2 features distributed each with a 2-Gaussian 
%with means mu1 & mu2 and isotropic covariances
mu1=[ones(N,1) 5*ones(N,1)];
mu2=[-5*ones(N,1) 1*ones(N,1)];
class1_std = 1;
class2_std = 1.1;

%generate class features and target labels
X = [class1_std*randn(N,2)+mu1;2*class2_std*randn(N,2)+mu2];
t = [ones(N,1);zeros(N,1)];

%Variance of prior
alpha=100;

%Define contour grid of weighting coefficients w1 & w2 for
%logistic regression model
[w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n,n]=size(w1);
W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

%Compute the log-prior and likelihood and joint likelihood
%at each point on grid defined above
f=W*X';
Log_Prior = log(gauss(zeros(1,D),eye(D).*alpha,W));
Log_Like = W*X'*t - sum(log(1+exp(f)),2); 
Log_Joint = Log_Like + Log_Prior;

%display contours of log-prio, likelihood and joint
figure
subplot(131)
contour(w1,w2,reshape(-Log_Prior,[n,n]),30);
title('Log-Prior');
subplot(132)
contour(w1,w2,reshape(-Log_Like,[n,n]),30);
title('Log-Likelihood');
subplot(133)
contour(w1,w2,reshape(-Log_Joint,[n,n]),30);
title('Log-Unnormalised Posterior')
hold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Identify the parameters w1 & w2 which maximise the posterior (joint)
[i,j]=max(Log_Joint);
plot(W(j,1),W(j,2),'.','MarkerSize',40);

%Newton routine to find MAP values of w1 & w2
%Fix number of steps to 10 and initial estimate to w1=0, w2=0
N_Steps = 10;
w = [0;0];

for m=1:N_Steps
    %store updated parameter values and plot evolution of estimates
    ww(m,:) = w;
    plot(ww(:,1),ww(:,2),'k.-');
    drawnow
    pause(0.1)
    
    %Newton Step
    P = 1./(1 + exp(-X*w));
    A = diag(P.*(1-P));
    H = inv(X'*A*X + eye(D)./alpha);
    w = H*X'*(A*X*w + t - P);

    %Compute new likelihood and unormalised posterior values
    f=X*w;
    lpr = log(gauss(zeros(1,D),eye(D).*alpha,w'));
    llk = f'*t - sum(log(1+exp(f))); 
    ljt = llk + lpr;
    fprintf('Log-Likelihood = %f, Joint-Likelihood = %f\n',llk,ljt)
end

fprintf('Maximum of Joint Likelihood = (%f, %f)\n',W(j,1),W(j,2));
fprintf('Estimate of Maximum of Joint Likelihood = (%f, %f)\n',w(1),w(2));

%plot the data points from both classes and show the contour of P(C=1|x)
figure
Posterior = 1./(1+exp(-W*w));
contour(w1,w2,reshape(Posterior,[n,n]),30);
hold on
plot(X(find(t==1),1),X(find(t==1),2),'r.');
plot(X(find(t==0),1),X(find(t==0),2),'bo');

%plot the data points and the values of P(C=1|x)=0.5 i.e. the separating
%plane which distinguishes both classes
figure
contour(w1,w2,reshape(Posterior,[n,n]),[0.5 0.5]);
hold on
plot(X(find(t==1),1),X(find(t==1),2),'r.');
plot(X(find(t==0),1),X(find(t==0),2),'bo');

%Compute the Laplace Approximation

%Numerically compute normalising constant for posterior by
%drawing 10000 samples from the prior
Nsamps = 10000;
Wsamp = alpha*randn(Nsamps,D);
f=Wsamp*X';
pr = gauss(zeros(1,D),eye(D).*alpha,Wsamp);
lk = exp(Wsamp*X'*t - sum(log(1+exp(f)),2)); 
Z = sum(lk.*pr);

%Show contour plots of the Posterior and the Laplace approximation to the
%posterior
Log_Laplace_Posterior = log(gauss(w',H,W)+SMALL_NOS); %to prevent zero-log
figure
subplot(121)
contour(w1,w2,reshape(-Log_Joint + log(Z),[n,n]),30);
hold;
plot(W(j,1),W(j,2),'.','MarkerSize',40);
title('Log Posterior')
subplot(122)
contour(w1,w2,reshape(-Log_Laplace_Posterior,[n,n]),30);
hold
plot(W(j,1),W(j,2),'.','MarkerSize',40);
title('Laplace Approximation to Posterior')


