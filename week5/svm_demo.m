clear
Step=0.5;
N = 50;
C=1000;
X = [randn(N/2,2);randn(N/2,2)+[ones(N/2,1).*6 zeros(N/2,1)]];
t=[ones(N/2,1);-ones(N/2,1)];
[alpha,w_0,alpha_index]=monqp0(diag(t)*X*X'*diag(t),ones(N,1),t,C,1e-6);

%Define contour grid 
mn = min(X);
mx = max(X);
[x1,x2]=meshgrid(floor(mn(1)):Step:ceil(mx(1)),floor(mn(2)):Step:ceil(mx(2)));
[n11,n12]=size(x1);
[n21,n22]=size(x2);
XG=[reshape(x1,n11*n12,1) reshape(x2,n21*n22,1)];

f = (t(alpha_index).*alpha)'*X(alpha_index,:)*XG' + w_0;

plot(X(alpha_index,1),X(alpha_index,2),'go');
hold
plot(X(1:N/2,1),X(1:N/2,2),'.')
plot(X(N/2+1:N,1),X(N/2+1:N,2),'r.')
contour(x1,x2,reshape(f,[n11,n12]),[0 0]);
hold off