%week 8 demo 2
%This script demonstrates a situation where standard K-means will fail when
%the two distinct clusters of data share the same mean value - this is
%achieved by making one cluster of data such that points are uniformly
%distributed within an annulus which is centered at the point (0,0). The
%sceond cluster corresponds to data which has an isotropic Gaussian 
%distribution centered at (0,0) and whose variance is sufficiently small
%that points in this cluster are distinct from those within the annulus.

clear
load wk8_demo_dat;
%Run standard K-means clustering assuming K = 2 - the true value
[H,j,e]=kmeans(X,2,30);
subplot(121)
plot(X(find(j==1),1),X(find(j==1),2),'.');
hold
plot(X(find(j==2),1),X(find(j==2),2),'ro');
title('K-Means Clustering');

%Run Kernel K-means assuming K = 2 AND the parameter of the kernel (width
%for an RBF is also passed - clearly this has to be selected in some
%reasonable way - cross validation is a practical way to achieve this.
[j,e] = kernel_kmeans(X,2,30,1);
subplot(122)
plot(X(find(j==1),1),X(find(j==1),2),'.');
hold
plot(X(find(j==2),1),X(find(j==2),2),'ro');
title('Kernel K-Means Clustering');

