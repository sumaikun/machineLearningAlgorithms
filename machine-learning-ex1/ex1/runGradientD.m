X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
%theta = zeros(2, 1); % initialize fitting parameters
num_iters = 1500;
alpha = 0.02;

[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

visualizeP()

hold on;

h = X * theta + 1

plot(X,h)