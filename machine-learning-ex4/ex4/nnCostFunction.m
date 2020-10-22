function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));% 25X401
Theta2_grad = zeros(size(Theta2));% 10X26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can veriyk that your
%         cost function computation is correct by veriyking the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

errors = zeros(1,num_labels)% 1*10
%X = [ones(m, 1) X];  % 5000 * 401
X1 = [ones(m, 1) X];
% X is a1, t1 is a2 and t2 is a3    
t1 = sigmoid(X1 * Theta1'); %  5000 * 401 X 401 * 25 = 5000 * 25 
t1 = [ones(m, 1) t1]; % 5000 * 26
t2 = sigmoid( t1 * Theta2'); %  5000 * 26 X 26 * 10 = 5000 * 10
%apply one vs all
%for t2 in this version for each column must exist a vector that represents the desire value and this vector must be  filled of ones
%because the sample size is 5000 an the options to predicts are then the matrix that multiply the neurons for cost function must be vector of
% m * numlabels 

yk = zeros(size(y,1),num_labels) % 5000*10

for i = 1:m
    yk(i,y(i)) = 1;
end


%chaging y by yk reduce a lot the errors, yk is different from y because it must be a matrix two for the two iterations

errors = ( -yk .* log( t2 ) ) - ( 1 - yk ) .* log( 1 - t2 )  % here is not matrix multiplication it is point multiplication in this way the zero values reduce

%thye complexity of neural results

rp =  lambda/(2*m) * (  sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) ) % this result on a scalar

J = 1/m * sum(sum( errors )) + rp  % this has to be a scalar, first sum turn a 10*10 matrix on a 1*10 vector

% -------------------------------------------------------------
%backpropagation
% =========================================================================

for t = 1 : m    

    %The first values are a gradients initialize with a non zero values

    a1 = [1 X(t,:)]'; % 401*1
    z2 = Theta1 * a1; % 25 * 401 X 401 * 1
    a2 = sigmoid(z2); 
    a2 = [1; a2]; % 26 * 1
    z3 = Theta2 * a2; % 10 * 26 X 26 * 1
    a3 = sigmoid(z3);
 
    z2 = [1; z2];% 26 * 1

    %k represents the number of labels in this case is the number from 0 to 9 so are 10 labels

    delta3 = a3 - yk'(:, t); % 10 * 1
    delta2 = (Theta2' * delta3) .* sigmoidGradient(z2); % 26*10 X 10*1
    delta2 = delta2(2:end);%25 *1
 
    Theta1_grad = Theta1_grad + delta2 * a1'; # 25*401 + 25*1 X 1*401
    Theta2_grad = Theta2_grad + delta3 * a2'; # 10*26 + 10*1 X 1*26
    

end

%unregularized gradients

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) +   (lambda/m) * Theta1(:,2:end) 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) +   (lambda/m) * Theta2(:,2:end)

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
