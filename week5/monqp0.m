function [alpha, lambda, pos] = monqp0(H,b,c,C,l,verbose,X,ps)

% min 1/2  u' A u - b' u                  u c'est alpha
%  u
% contrainte   c' u = a                     y'*alpha = 0
%
% et         0 <= u <= C


%--------------------------------------------------------------------------
%                                verifications
%--------------------------------------------------------------------------
[n,d] = size(H);
[nl,nc] = size(b);
[nlc,ncc] = size(c);
if d ~= n
	error('H must be a squre matrix n by n');
end
if nl ~= n
	error('H and b must have the same number of row');
end

if nlc ~= n
	error('H and c must have the same number of row');
end
if ncc ~= 1
	error('c must be a row vector');
end
if nc ~= 1
	error('b must be a row vector');
end

if nargin < 5		% default value for the regularisation parameter
	l = 0;				
end;
if nargin < 4		% default value for the uper bound
	C = 100;				
end;

if nargin < 6		% default value for the display parameter
	verbose = 0;				
end;


fid = 1; %default value, curent matlab window
%--------------------------------------------------------------------------


nsup = n;
I = nsup*eye(nsup);

M = [(H+l*I)  c ;  c'  0 ];
Minit = M;
B = [b;0];
Binit = B;
cinit = c;

ind  = 1;indout=[];indd = (1:n);pos = indd;

stop = 0;
alpha = [];
Msup = [];
indsuptot = [];	
	
%--------------------------------------------------------------------------
%                       I N I T I A L I S A T I O N 
%--------------------------------------------------------------------------
 
%k = 2;										% c in (-1,1) and Y in (1,2)
%[A]=knnd(X,X,(c+3)/2,k);			
%malclasses = find(A~=((c+3)/2));
%bienclasses = find(A==((c+3)/2)); 		% keyboard

bienclasses = (2:n);     
malclasses = 1;;

% keyboard 
 
if length(malclasses) == 0
	disp('y''a pas de mal classes, c''est louche'); % quoi faire ???
end;

alpha = zeros(n,1);  
alpha(malclasses) = C/2;

nsup = length(malclasses);

% disp('la contrainte : nb active'); disp([c'*alpha nsup]);
alpha(bienclasses) = [];
M(bienclasses,:) = [];
M(:,bienclasses) = [];
B(bienclasses) = [];                 %  keyboard
pos = malclasses;
indd = 0*indd;
indd(malclasses) = malclasses;

%--------------------------------------------------------------------------
%                            M A I N   L O O P
%--------------------------------------------------------------------------
Jold = 10000000000000000000; sol = 0;
if verbose ~= 0
  disp('      Cost     Delta Cost  #support  #up saturate');
  nbverbose = 0;
end

while stop == 0

        I =  nsup*eye(nsup);
	[nn,mm] = size(Msup);         
	Un = ones(mm,1); 
        lambdaA = sol(length(sol)) ;
                                              % keyboard;
[J,yx] = cout(H,alpha,cinit,C,indd);
if verbose ~= 0
  nbverbose = nbverbose+1;
  if nbverbose == 20
  disp('      Cost     Delta Cost  #support  #up saturate');
  nbverbose = 0;
end
if Jold == 0
   fprintf(fid,'| %11.2f | %8.4f | %6.0f | %6.0f |\n',[J (Jold-J) nsup length(indsuptot)]);
   else
fprintf(fid,'| %11.2f | %8.4f | %6.0f | %6.0f |\n',[J (Jold-J)/abs(Jold) nsup length(indsuptot)]);
end
end
Jold = J;

	if isempty(Msup)
              sol = M \ B;
	else
              sol = M \ (B-[C*Msup*Un ; C*sum(cinit(indsuptot))]);
	end
	alphaNew = sol(1:length(sol)-1);      % il ne faut pas supprimer
                                              % la derniere variable

if  min(alphaNew)<0 | max(alphaNew)>C 
        indinf = find( alphaNew < 0);
	indsup = find( alphaNew > C);

	d = alphaNew - alpha;	              % direction de descente

        [tI indI] = min(-alpha(indinf)./d(indinf));    % pas de descente
        [tS indS] = min((C-alpha(indsup))./d(indsup)); % pas de descente
	if isempty(tI) , tI = tS + 1; end;
	if isempty(tS) , tS = tI + 1; end;
	t = min(tI,tS);
                                              %  keyboard;
        alpha  = alpha + t * d;

	pos = find(indd > 0);		% indices des variables actives
%	disp('ca descend - main loop');        % 
        if t == tI
		ind = indinf(indI);
                if ~isempty(Msup), Msup(ind,:) = []; end;   
% 	        disp(' on enleve une petite '); 
%	        disp([indd(pos(ind)) length(alphaNew)]);
	        indd(pos(ind)) = 0;
	else															%	keyboard
		ind = indsup(indS);
                indsuptot = [indsuptot ; pos(ind)];
	        Msup = Minit(pos,indsuptot);	% contraintes sup saturees
                Msup(ind,:) = [];
% 	        disp(' on enleve une grosse ');
%	        disp([indd(pos(ind)) length(alphaNew)]);
				indd(pos(ind)) = -1;                       

    end
	M(:,ind) = [];
	M(ind,:) = [];
	alpha(ind) = [];
	B(ind) = [];

        nsup = nsup - 1;
	pos(ind) = [];

else
	soltot = zeros(n,1);
	posok = find(indd > 0);
	posA = find(indd==0);			% liste des contriantes saturees
	posB = find(indd==-1);			% liste des contriantes saturees
                                                % keyboard
	soltot(posok) = alphaNew;
	soltot(posB) = C;
      				                % multiplicateurs de lagrange
	lamb = (H*soltot  - b) + c*sol(nsup+1); %    
                                                % ; % disp(lamb');
      if  min(lamb(posA)) < -sqrt(eps)	|  min(-lamb(posB)) <  -sqrt(eps) % on ajoute une containte.

          if  isempty(posB) | min(lamb(posA)) < min(-lamb(posB))  
		  [minlam lampos] = min(lamb(posA));
	          lampos = posA(lampos);
%  	          disp('ca monte - une petite');                       
                  camonte = 'petit';
             else
		  [minlam lampos] = min(-lamb(posB));        
                  lampos = posB(lampos);
% 	          disp('ca monte - une grosse');           %     keyboard      
                   camonte = 'grose'; 
                  aaa = find(indsuptot == lampos);
		  indsuptot(aaa) = [];                                
                  Msup(:,aaa) = [];
         end
%disp('indsuptot')
%disp(indsuptot')
%	  disp([lampos  length(alphaNew)]);                   % 
          
          inserpos = max(find(posok < lampos));
		if isempty(inserpos) inserpos=0; end;
 	  indd(lampos) = lampos;                              %    

          M = [M(:,1:inserpos) , Minit([posok n+1],lampos) , M(:,inserpos+1:nsup+1)];
	  B = [B(1:inserpos);Binit(lampos);B(inserpos+1:nsup+1)];
	  if camonte == 'petit' 
             alpha = [alphaNew(1:inserpos);0;alphaNew(inserpos+1:nsup)];
            else
             alpha = [alphaNew(1:inserpos);C;alphaNew(inserpos+1:nsup)];
          end
          pos = [posok(1:inserpos) lampos posok(inserpos+1:nsup)];
          M = [M(1:inserpos,:) ; Minit(lampos,[pos n+1]) ; M(inserpos+1:nsup+1,:)];  

          if  isempty(indsuptot)
              Msup = []; 
          elseif  isempty(Msup)
              Msup =  Minit(lampos,indsuptot) ;
          else
              Msup = [Msup(1:inserpos,:) ; Minit(lampos,indsuptot) ; Msup(inserpos+1:nsup,:)];
	  end;

	  nsup = nsup + 1;
	else
		stop = 1;	              % on est bien au minimum...
	end
end
%--------------------------------------------------------------------------
%                        E N D   M A I N   L O O P
%--------------------------------------------------------------------------

end


lambda = sol(length(alphaNew)+1);
alpha = alphaNew(1:nsup);               % disp(indd);
pos = pos(1:nsup);

if ~isempty(indsuptot)
	alpha = zeros(n,1);
	posok = find(indd > 0);
	posA = find(indd==0);			% liste des contriantes saturees
	posB = find(indd==-1);			% liste des contriantes saturees
                                                % keyboard
	alpha(posok) = alphaNew;
	alpha(posB) = C;
	pos = sort([posok posB]);
	alpha(posA) = [];
	
 end
