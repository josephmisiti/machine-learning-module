clear
Step=0.5;
polypower = 3;
width = 1;


X=load('rip_dat_tr.txt');
Xt=load('rip_dat_te.txt');
t=2.*X(:,3) - 1;
X(:,3)=[];
tt=2.*Xt(:,3) - 1;
Xt(:,3)=[];
N = size(X,1);
Nt = size(Xt,1);
Err = [];

for C = 1:20
    K = kernel_func(X,X,'poly',width,polypower);
    [alpha,w_0,alpha_index]=monqp0(diag(t)*K*diag(t),ones(N,1),t,C,1e-6);
    Kt = kernel_func(X(alpha_index,:),Xt,'poly',width,polypower);
    ft = sign((t(alpha_index).*alpha)'*Kt + w_0);
    err=100 - 100*sum(ft' == tt)/Nt;
    Err=[Err;err];
    fprintf('Soft Margin Parameter = %f; Test Error = %f\n',C, err);    
end
figure 
subplot(121)
plot(Err);
[Cmin,ErrMin]=min(Err);
[alpha,w_0,alpha_index]=monqp0(diag(t)*K*diag(t),ones(N,1),t,Cmin,1e-6);

%Define contour grid 
mn = min(X);
mx = max(X);
[x1,x2]=meshgrid(floor(mn(1)):Step:ceil(mx(1)),floor(mn(2)):Step:ceil(mx(2)));
[n11,n12]=size(x1);
[n21,n22]=size(x2);
XG=[reshape(x1,n11*n12,1) reshape(x2,n21*n22,1)];
KG = kernel_func(X(alpha_index,:),XG,'poly',width,polypower);
f = (t(alpha_index).*alpha)'*KG + w_0;
subplot(122)
plot(X(alpha_index,1),X(alpha_index,2),'go');
hold
plot(X(1:N/2,1),X(1:N/2,2),'.')
plot(X(N/2+1:N,1),X(N/2+1:N,2),'r.')
contour(x1,x2,reshape(f,[n11,n12]),[0 0]);
hold off