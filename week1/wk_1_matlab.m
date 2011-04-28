%Machine Learning Module
%Week One Laboratory Exercise

x=load('long_jump_data.txt');

x_train = x(1:20,1);
t_train = x(1:20,2);
x_test = x(21:end,1);
t_test = x(21:end,2);

Polynomial_Order = 4;
X_train = [];
X_test = [];

for i = 0:Polynomial_Order
    X_train = [X_train x_train.^i];
    X_test = [X_test x_test.^i];

    w_hat = inv(X_train'*X_train)*X_train'*t_train;

    t_train_hat = X_train*w_hat;
    t_test_hat = X_test*w_hat;
    mse_train(i+1) = mean((t_train - t_train_hat).^2);
    mse_test(i+1) = mean((t_test - t_test_hat).^2);
end

subplot(2,2,1);
plot(0:Polynomial_Order,mse_train,'dg-');
title('Train Error');

subplot(2,2,2);
plot(0:Polynomial_Order,mse_test,'dr-');
title('Test Error');

[min_test_val,min_test_index] = min(mse_test);
[min_train_val,min_train_index] = min(mse_train);

t_train_hat_min = X_train(:,1:min_train_index)*...
inv(X_train(:,1:min_train_index)'*X_train(:,1:min_train_index))*...
X_train(:,1:min_train_index)'*t_train;

t_test_hat_min = X_train(:,1:min_test_index)*...
inv(X_train(:,1:min_test_index)'*X_train(:,1:min_test_index))*...
X_train(:,1:min_test_index)'*t_train;

subplot(2,2,3)
plot(x_train, t_train,'og');
hold on;
plot(x_train,t_train_hat_min);
title('Minimum Train Error Model');

subplot(2,2,4)
plot(x_train, t_train,'og');
hold on;
plot(x_train,t_test_hat_min);
title('Minimum Test Error Model');