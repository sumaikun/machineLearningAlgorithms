function J = costFunction(X, y, theta)
    m = size(X,1);
    predictions = X * theta;
    sqlErrors = (predictions - y).^2;
    J = 1/(2*m) * sum(sqlErrors);
end
