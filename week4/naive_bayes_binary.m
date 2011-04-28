%Demo of Naive Bayes Classifier for binary data
%using the 4 class subset of 20newsgroups with 100 terms
clear
load 20news_w100;
X = documents';     
t = full(newsgroups');  

Nclasses = full(max(t));
train_test_frac = 0.60;
[N,D] = size(X);

I = randperm(N);
Ntrain = floor(N*train_test_frac);
Ntest = N - floor(N*train_test_frac);

fprintf('Random Partition of %d Train Samples, %d Test Samples\n', Ntrain, Ntest);

Xtrain = X(I(1:Ntrain),:);
ttrain = t(I(1:Ntrain));

Xtest = X(I(Ntrain+1:end),:);
ttest = t(I(Ntrain+1:end));

%Naive Bayes Classifier for general Nclasses
for c=1:Nclasses
    Prior(c) = mean((ttrain==c));
    p(c,:) = (sum(Xtrain(find(ttrain==c),:)) + 1)./(length(find(ttrain==c)) + 2);
    Log_Like = sum((Xtest.*log(repmat(p(c,:),Ntest,1))),2) +...
               sum((1-Xtest).*log((1-repmat(p(c,:),Ntest,1))),2);
    Log_Posterior(:,c) = Log_Like + log(Prior(c)).*ones(Ntest,1);
end

Posterior = exp(Log_Posterior);
Posterior = Posterior./repmat(sum(Posterior,2),1,Nclasses);

[i,j]=sort(ttest);
for k=1:Nclasses
    subplot(Nclasses,1,k)
    plot(Posterior(j,k));
    title(sprintf('Predictive Posterior %s for News Groups',groupnames{k}));
    fprintf('Class %d  = %s\n', k, groupnames{k});
end

[max_post, t_pred] = max(Posterior,[],2);
fprintf('Percentage Predictions Correct %f\n',100*mean(t_pred == ttest));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some fun stuff
% List the top ten most probable words in each class
MaxI = 5;
for k=1:Nclasses
    [most_probable_words,index_mpw]=sort( -p(k,:) );
    fprintf('Most probable words in Class %s\n',groupnames{k});
    wordlist(index_mpw(1:MaxI))'
    pause
    fprintf('Hit enter to continue...\n\n');
end








