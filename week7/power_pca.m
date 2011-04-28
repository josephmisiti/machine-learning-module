function [B,D]=power_pca(C)
%A little routine to compute PCA given a covariance matrix C
N = size(C,1);

threshold = 1e-3;
Max_Its = 1000;

%loop round all dimensions of the covariance matrix
for n=1:N
    
   %initialise the principal eigenvector and set norm to unity 
   x = randn(N,1);
   y = x./sqrt(x'*x);
   
   %monitor convergence 
   err = 1e20;
   its = 1;
   
   %main loop to compute single eigenvector
   while (err > threshold) | (its < Max_Its)
       x = C*y;
       y_new = x./sqrt(x'*x);
       
       err = sum((y_new - y).^2);
       y = y_new;
       
       %set eigenvalue
       D(n) = sqrt(x'*x);
       
       %increment counter
       its = its + 1;
   end
   
   %set the column vectors to be the found eigenvectors
   B(:,n) = y_new;
   
   %deflate the covariance matrix
   C = C - D(n)*y_new*y_new';   
end
D=diag(D);