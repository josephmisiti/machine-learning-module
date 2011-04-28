clear
%A small constant used to avoid log of zero problems
SMALL_NOS = 1e-200;

%Two hyperparameters of model
Polynomial_Order = 3;
alpha=100; 

%Load and prepare train & test data
X=load('rip_dat_tr.txt');
Xt=load('rip_dat_te.txt');
t=X(:,3);
X(:,3)=[];
tt=Xt(:,3);
Xt(:,3)=[];

%Plot two classes in train set
subplot(221)
plot(X(find(t==1),1),X(find(t==1),2),'r.');
hold
plot(X(find(t==0),1),X(find(t==0),2),'o');
title('Scatter Plot of Data from Classes');
fprintf('Two overlapping Non-Gaussian Class Distributions\n')
fprintf('Hit Enter to Continue....');
pause;

%Limits and grid size for contour plotting
Range=1.3;
Step=0.1;
%Define contour grid 
[w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n,n]=size(w1);
W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

%Create Polynomial Basis
XX = []; XXt = []; WW = [];
for i = 0:Polynomial_Order
    XX = [XX X.^i];
    XXt = [XXt Xt.^i];
    WW = [WW W.^i];
end
[N,D] = size(XX);
Nt = size(XXt,1);

%Newton routine to find MAP values of w
%Fix number of steps to 20 and initial estimate to w=0
N_Steps = 10;
w = zeros(D,1);

for m=1:N_Steps    
    %Newton Step
    P = 1./(1 + exp(-XX*w));
    A = diag(P.*(1-P));
    H = inv(XX'*A*XX + eye(D)./alpha);
    w = H*XX'*(A*XX*w + t - P);

    %Compute new likelihood and unormalised posterior values
    f=XX*w; % train
    ft=XXt*w; %test
    lpr = log(gauss(zeros(1,D),eye(D).*alpha,w'));
    llk = f'*t - sum(log(1+exp(f))); %training likelihood
    ljt = llk + lpr;
    fprintf('Log-Likelihood = %f, Joint-Likelihood = %f\n',llk,ljt)
end

%Compute Overall performance
%Train 
Train_Like = llk;
Test_Like = ft'*tt - sum(log(1+exp(ft)));
Train_Error = 100 - 100*sum( (1./(1+exp(-XX*w)) > 0.5) == t)/N; %number of miss-classifications
Test_Error = 100 - 100*sum( (1./(1+exp(-XXt*w)) > 0.5) == tt)/Nt;
fprintf('\n\nClassifier Performance Statistics using MAP Value\n');
fprintf('Training Likelihood = %f, Training 0-1 Error = %f\n',Train_Like,Train_Error);
fprintf('Test Likelihood = %f, Test 0-1 Error = %f\n',Test_Like,Test_Error);

%plot the data points from both classes and show the contour of P(C=1|x)
subplot(222)
Posterior = 1./(1+exp(-WW*w));
contour(w1,w2,reshape(Posterior,[n,n]));
hold on
plot(X(find(t==1),1),X(find(t==1),2),'r.');
plot(X(find(t==0),1),X(find(t==0),2),'bo');
title('Contour of Posterior P(C=1|x)');

%plot the data points and the values of P(C=1|x)=0.5 i.e. the separating
%plane which distinguishes both classes
subplot(223)
contour(w1,w2,reshape(Posterior,[n,n]),[0.5 0.5]);
hold on
plot(X(find(t==1),1),X(find(t==1),2),'r.');
plot(X(find(t==0),1),X(find(t==0),2),'bo');
title('Decision Boundary P(C=1|x) = 0.5')

subplot(224)
bar(w./sqrt(diag(H)))
title('Hessian Normalised Basis Weighting Coefficients')




