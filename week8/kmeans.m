function [M,j,e] = kmeans(X,K,Max_Its)

%This is a simple and naive implementation of the standard K-Means
%clustering algorithm for the Machine Learning II course.

%The data matrix X (N x D) is passed as argument
%The number of cluster K is passed as argument
%The maximum nos of iterations Max_Its is passed as argument

%The function returns the matrix M (K x D) - corresponding to the K mean
%vector values

%The functin returns the clusters which each point has been allocated to
%1.. K in the vector j.

[N,D]=size(X);  %N - nos of data points, D dimension of data
I=randperm(N);  %a random permutation of integers 1:N - required 
                %to set initial mean values

M=X(I(1:K),:);  %M is the initial K x D matrix of mean values - 
                %simply setting to values of K randomly selected data
                %points
Mo = M;         

for n=1:Max_Its
    %Create distance matrix which is N x K indicating distance that each data
    %point is from the current mean values (of which there are K)
    for k=1:K
        Dist(:,k) = sum((X - repmat(M(k,:),N,1)).^2,2);
    end

    %No we simply find which of the K-mean each data point is nearest to -
    %so we find the minimum distance of K for each data point. This
    %operation can be easily achieved in one line of Matlab using the min function. 
    [i,j]=min(Dist,[],2);
    
    %Now that we have the new allocations of points to clusters based on
    %the minimum distances obtained form the previous operation we can
    %revise our estimates of the position of each mean vector by simply
    %taking the mean vlue of all points which have been allocated to each
    %cluster using the folowing simple routine.
    
    for k=1:K
        if size(find(j==k))>0
            M(k,:) = mean(X(find(j==k),:));
        end
    end
    
    %we create an N x K dimensional indictor matrix - each row will have a
    %1 in the column corresponding to the cluster that the data point (row)
    %has been allocated to - this is really only required to compute the
    %overall error assocated with the current partitioning.
    
    Z = zeros(N,K);
    for m=1:N
        Z(m,j(m)) = 1;
    end
    
    %This simply prints the current value of the error criterion which
    %K-means is trying to minimise.
    e = sum(sum(Z.*Dist)./N);
    fprintf('%d Error = %f\n', n, e);
    Mo = M;
end