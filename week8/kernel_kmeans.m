function [z,e] = kernel_kmeans(X,K,Max_Its,kwidth)
%This is a simple implementation of Kernel K-means clustering - an
%interesting paper which proposed kernel based Kmeans clustering is [1]
%Girolami, M, Mercer Kernel-Based Clustering in Feature Space, 
%IEEE Trans Neural Networks, 13(3),780 - 784, 2002. 


%Create the kernel matrix.
[N,D]=size(X);
C = kernel_func(X,X,'gauss',kwidth,1);

%initialise the indictaor matrix to a random segmentation of the data
Z = zeros(N,K);
for n = 1:N
  Z(n,rand_int(K)) = 1; 
end

%main loop
for its = 1:Max_Its
    %compute the similarity of each data point to each cluster mean in
    %feature space - note we do not need to compute store or update a mean
    %vector s we are using the kernel-trick - cool eh?
    for k=1:K
        Nk = sum(Z(:,k));
        Y(:,k) = diag(C) - 2*C*Z(:,k)./Nk + Z(:,k)'*C*Z(:,k)./(Nk^2);
    end

    %Now we find the cluster assignment for each point based on the minimum
    %distance of the point from the mean centres in feature space using the
    %Y matrix of dissimilarities
    [i,j]=min(Y,[],2);
    
    %this simply updates the indictor matrix Z refleting the new
    %allocations of data points to clusters
    Z = zeros(N,K);
    for n=1:N
        Z(n,j(n)) = 1;
    end
    
    %compoute the verall error
    e = sum(sum(Z.*Y))./N;
    fprintf('%d Error = %f\n', its, e);
end

%return the clutsers that each data point has been allocated to
for n=1:N
    z(n) = find(Z(n,:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this is a little utility functino which returns a random integer between 1
%& Max_Int.
function u = rand_int(Max_Int)
u=ceil(Max_Int*rand);