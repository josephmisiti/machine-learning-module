%This is a demo of the maximum likelihood estimates of
%the parameters of a 2-D Gaussian from a Finite sample of data
%the assumption that the parametric form of the density is Gaussian
%is of course well justified in this illustrative example

function L = gauss_density_est(N)
%Set true mean and covariance for Gaussian
m = [1 3]';
C = [1.5 0.6;0.6 0.4];
L=[];

%Create a grid of points on which to compute the density
[x1,x2]=meshgrid(-3:0.1:6,-3:0.1:6);
[n,n]=size(x1);
XX=[reshape(x1,n*n,1) reshape(x2,n*n,1)];
        
%Generate a sample of N points drawn from the true Gaussian density
for i=1:N
    X(i,:) = multi_var_gauss_sampler(m,C)';
end

%Make maximum likelihood estimates of the paramters
m_hat = mean(X);
C_hat = cov(X);
%compute the probability density of the data sample under the 2-D Gaussian
%with the ML parameter estimates 
p=gauss(m_hat',C_hat,X);
Log_Like=mean(log(p));


%Nice picture of the data sample and the isocontours of probability density
subplot(121)
plot(X(:,1),X(:,2),'.');
hold on;
pt=gauss(m_hat',C_hat,XX);
contour(x1,x2,reshape(pt,[n,n]));
subplot(122)
contour(x1,x2,reshape(pt,[n,n]));
hold on;
%compute true density
pt=gauss(m',C,XX);
contour(x1,x2,reshape(pt,[n,n]));
hold off;

%Report the true and estimated parameter values
fprintf('True Mean [%f %f]\n',m(1),m(2));
fprintf('Est  Mean [%f %f]\n\n',m_hat(1),m_hat(2));
fprintf('True Covariance [%f %f;%f %f]\n',C(1,1),C(1,2),C(2,1),C(2,2));
fprintf('Est  Covariance [%f %f;%f %f]\n\n',C_hat(1,1),C_hat(1,2),C_hat(2,1),C_hat(2,2));
fprintf('Log Likelihood Score = %f\n',Log_Like);
