%This demo uses the function svd which is closely related to PCA in that it
%provides a decomposition of a matrix 
%X = USV' and so X'X = VSU'USV' = VS^2V' where V is the matrix whose
%columns are the eigenvectors of X'X i.e a scaled version of the principal
%directions and the matrix U has columns which are the principal components
%or the projections of X onto V i.e. U = XVinv(S)

clear
load('olivettifaces.mat');
Selected_Face = 125;%312;%255;

X=faces';
[N_faces,D]=size(X);
mean_face = mean(X);
X = X - repmat(mean_face,N_faces,1);
fprintf('Performing PCA.... stay tuned\n');
[U,S,V]=svd(X);
subplot(131)
imagesc(reshape(X(Selected_Face,:)+mean_face,sqrt(D),sqrt(D)));
title('Original Image');

recon_err=[];

for i=1:N_faces
    X_Reconst=U(Selected_Face,1:i)*S(1:i,1:i)*V(:,1:i)' + mean_face;
    subplot(132)
    imagesc(reshape(X_Reconst',sqrt(D),sqrt(D)));
    title('Reconstructed Image');drawnow;
    recon_err = [recon_err;sqrt(mean((X_Reconst - (X(Selected_Face,:) + mean_face) ).^2,2))];
    colormap gray
    subplot(133)
    plot(1:i,recon_err,'LineWidth',3);
    title('Reconstruction Error');
    pause(0.1)
    fprintf('%d:Reconstruction Error = %f\n',i,recon_err(i))
end


