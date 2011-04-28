%week_8_demo_1
%images segmentation. This is a rather nice demo which shows how clustering
%methods can be empoyed in segmenting images into self similar regions


%segmenting a gray scale image of a face - achieved by clustering each of
%the pixel values based on gray level
load('olivettifaces.mat');
[H,j]=kmeans(faces(:,1),5,10);

figure
colormap gray
subplot(121)
imagesc(reshape(faces(:,1),64,64));drawnow
subplot(122)
imagesc(reshape(j,64,64));drawnow


%segmenting a jpg image of water lillies - the image is represented such that 
%each pixel is rperesented as a three dimensional vector in RGB space so we
%perform pixel clustering based on color values - this is quite a nice
%image which demonstrates that the leaves and flowers of water lillies can
%be separated from each other and segmented from the background - this is
%due to the uniform colors across each of the leaves and flowers.
clear
X = imread('water_lillies.jpg','jpg');
A = [double(reshape(X(:,:,1),600*800,1))...
    double(reshape(X(:,:,2),600*800,1))...  
    double(reshape(X(:,:,3),600*800,1))];
[H,j,e]=kmeans(A,3,10);
figure
subplot(121)
imagesc(X);drawnow
subplot(122)
imagesc(reshape(j,600,800));drawnow


%this is another vey nice example as the dog, water, grass & road can be
%segmented. However this also shows the variability in the solutions
%obtained - for a single run you may or may not get a good segmentation
%into each of the regions of interest. So in the following loop K-means is
%run mutiple times toring the segmentation which yields the smallest error
%- which should correspond to the best segmentation.
clear
X = imread('wee_dog.jpg','jpg');
X=(X(15:end-15,:,:)); %crop image
A = [double(reshape(X(:,:,1),71*100,1))...
    double(reshape(X(:,:,2),71*100,1))...
    double(reshape(X(:,:,3),71*100,1))];
[H,j,e]=kmeans(A,4,20);
figure
subplot(121)
imagesc(X);drawnow
subplot(122)
imagesc(reshape(j,71,100));drawnow

%here we run the K-mean alorithm on the images of the wee dog five hundred
%times. We retain only the segmentation yielding the smallest value of
%error and also look at the distributuino of the error achieved - quite
%interesting.

A = A - repmat(mean(A),size(A,1),1);
A = A./repmat(std(A),size(A,1),1);
E=[];
emin =1e100;
for i=1:100
    [H,j,e]=kmeans(A,4,20);
    if e < emin
        j_min = j;
    end
    E=[E;e];
end
figure
subplot(121)
hist(E)
subplot(122)
imagesc(reshape(j_min,71,100));drawnow
    



