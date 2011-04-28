function K = kernel_func(X1,X2,kernel_,T,p)

[N1 d]		= size(X1);
[N2 d]		= size(X2);

switch kernel_

case 'gauss',
  K	= exp(-distSqrd(X1,X2,T));  

case 'poly',
  K	= (1+X1*T*X2').^p;
end
  
function D2=distSqrd(X,Y,T)
nx	= size(X,1);
ny	= size(Y,1);

D2	= sum((X.^2)*T,2)*ones(1,ny) + ones(nx,1)*sum((Y.^2)*T,2)' - 2*(X*T*Y');