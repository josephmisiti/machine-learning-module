function [J,lam] = cout(H,x,y,C,ind)


	[n,m] = size(H);
	X = zeros(n,1);
	posok = find(ind > 0);
	posA = find(ind==0);			% liste des contriantes saturees
	posB = find(ind==-1);			% liste des contriantes saturees
						%  keyboard                                               
	X(posok) = x;
	X(posB) = C;

	J = 0.5 *X'*H*X  - sum(X); 		 
				                % 0 normalement
 % keyboard
  
  lam = y'*X;
  
  
