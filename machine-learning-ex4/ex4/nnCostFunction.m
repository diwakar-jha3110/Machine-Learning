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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];
theta_1_n = Theta1;
theta_1_n(1)=0;
h_thetaX = sigmoid(X*Theta1');
h_thetaX = [ones(m, 1) h_thetaX];
h_thetaX_2 = sigmoid(h_thetaX*Theta2');
%[a,b]=size(h_thetaX_2)% [5000, 10]
%[a,b]=size(y) %[5000,1]
sqrerror = zeros(num_labels,1);
for i=1:num_labels
sqrerror(i) = (y'==i)*log(h_thetaX_2(:,i))+(1-(y'==i))*log(1- h_thetaX_2(:,i));
end;
theta2_n = Theta2;
theta_2_n(1)=0;
t_Theta1 = Theta1;
t_Theta1(:,1)=0;
t_Theta2 = Theta2;
t_Theta2(:,1)=0;
regulTurm = lambda*1/(2*m)*(sum(sum(t_Theta1.^2))+sum(sum(t_Theta2.^2)));
J= -1/m *sum(sqrerror)+ regulTurm ;

%part 2 
%[a,b] = size(Theta1) [25,401]
%[a,b] = size(Theta2) [10,26]
%[a,b] = size(X)   [5000,401]
%[a,b] = size(y) [5000,1]
% feedforward
h_thetaX_1 = sigmoid(X*Theta1'); 
%[a,b] = size(h_thetaX_1) [5000,25]
h_thetaX_1 = [ones(m, 1) h_thetaX_1];
%[a,b] = size(h_thetaX_1) [5000,26]

h_thetaX_2 = sigmoid(h_thetaX_1*Theta2'); 
%[a,b] = size(h_thetaX_2) [5000,10]
for i=1:num_labels
delta_3(i,:) = h_thetaX_2'(i,:) - (y'==i);  
end
%[a, b] = size(delta_3) [10, 5000]

%firstTrm = (Theta2'*delta_3);
%[a,b] = size(firstTrm) [26, 5000]

% secondturm = (h_thetaX_1.*(1-h_thetaX_1));
%[a,b] = size(secondturm) [5000,26]

delta_2 = (Theta2'*delta_3)'.* (h_thetaX_1.*(1-h_thetaX_1)) ;
%[a,b] = size(delta_2) [5000,26]

delta_2 = delta_2'(2:end,:); 
delta_1_acc = delta_2*X;
% [a,b] = size(delta_1_acc) [25,401]
delta_1_acc(:,2:end) = 1/m.*(delta_1_acc(:,2:end)+lambda*Theta1(:,2:end));
delta_1_acc(:,1) = 1/m.*(delta_1_acc(:,1));

delta_2_acc = delta_3 * (h_thetaX_1);
delta_2_acc(:,2:end) = 1/m.*(delta_2_acc(:,2:end)+lambda*Theta2(:,2:end)); 
delta_2_acc(:,1) = 1/m.*(delta_2_acc(:,1));
%[a,b] = size(delta_2_acc)[10,26]

Theta1_grad = delta_1_acc;
Theta2_grad = delta_2_acc;



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
