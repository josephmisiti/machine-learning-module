function [L,X] = gauss_density_est(N)
L=[];
%The data is drawn with equal probability from two Gaussian
%distributions with specific means and covariances as defined below
N2 = floor(N/2);
m1 = [0.5 2]';
m2 = [3 4]';
C1 = eye(2);
C2 = C1;

%Create a grod of points on which to compute the density
[x1,x2]=meshgrid(-3:0.1:6,-3:0.1:6);
[n,n]=size(x1);
XX=[reshape(x1,n*n,1) reshape(x2,n*n,1)];
  
%Generate a data sample with equal numbers of points drawn from both
%Gaussians
for i=1:N2
    X(i,:) = multi_var_gauss_sampler(m1,C1)';
    X(i+N2,:) = multi_var_gauss_sampler(m2,C2)';
end

%Estimate the mean and covariance of a single Gaussian - so you are
%assuming that a single parametric form of density is sufficient to
%faithfully represent the ubderlying data generating mechanism
m_hat = mean(X);
C_hat = cov(X);
p=gauss(m_hat',C_hat,X);
Gauss_Log_Like=mean(log(p));


%Estime the mean and covariance of a mixture of two Gaussians - we are
%assuming that we know which points were drawn from which Gaussian
m1_hat = mean(X(1:N2,:));
m2_hat = mean(X(N2+1:end,:));
C1_hat = cov(X(1:N2,:));
C2_hat = cov(X(N2+1:end,:));
p=0.5*gauss(m1_hat',C1_hat,X) + 0.5*gauss(m2_hat',C2_hat,X);
Mix_Gauss_Log_Like=mean(log(p));

%plot the data and density isocontours assuming a single Gaussian
subplot(121)
plot(X(:,1),X(:,2),'.');
hold on;
pt=gauss(m_hat',C_hat,XX);
contour(x1,x2,reshape(pt,[n,n]));
hold off;

%plot the data and density isocontours assuming two Gaussians generate the
%data
subplot(122)
plot(X(:,1),X(:,2),'.');
hold on;
pt=0.5*gauss(m1_hat',C1_hat,XX) + 0.5*gauss(m2_hat',C2_hat,XX);
contour(x1,x2,reshape(pt,[n,n]));
hold off;
%Comparison of average log-likelihood scores on TEST data from the grid under the two
%assumptions - the true semi-parametric form, as it si the true density, should yield a higher average
%log-likelihood
fprintf('Log Likelihood Score for Gaussian = %f\n',Gauss_Log_Like);
fprintf('Log Likelihood Score for Gaussian Mixture = %f\n',Mix_Gauss_Log_Like);

%Mesh plots
% figure
% subplot(121)
% pt=gauss(m_hat',C_hat,XX);
% mesh(x1,x2,reshape(pt,[n,n]));
% subplot(122)
% pt=0.5*gauss(m1_hat',C1_hat,XX) + 0.5*gauss(m2_hat',C2_hat,XX);
% mesh(x1,x2,reshape(pt,[n,n]));

