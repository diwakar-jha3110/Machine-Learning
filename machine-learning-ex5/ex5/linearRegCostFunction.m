function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%[a,b] = size(X)[12,2]
%[a,b] = size(theta)[2,1]
%[a,b] = size(y)[12,1]
J = 0;
grad = zeros(size(theta));
theta_1 = theta;
theta_1(1,:) = 0;
p = X*theta; % Prediction [12,1]
squareError = (p-y).^2;
j_theta_1 = 1/(2*m)*sum(squareError);
J= j_theta_1+ (lambda/(2*m))*sum(theta_1.^2);

grad_0 =  (1/m)*sum(X'(1,:)*(p-y)); % X'(1,:)
grad =  (1/m)*(X'*(p-y))+(lambda/m)*theta; % X'(2:end,:)
%(lambda/m).*theta_1
%(1/m)*sum(X'(2,:)*(p-y))
%X'
%p-y
%(1/m)*(X'*(p-y))
%m
grad(1)= grad_0(1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%==================== completed code
%J = 0;
%grad = zeros(size(theta));
%theta_1 = theta;
%theta_1(1) = 0;
%p = X*theta; % Prediction [12,1]
%squareError = (p-y).^2;
%j_theta_1 = 1/(2*m)*sum(squareError);
%J= j_theta_1+ (lambda/(2*m))*sum(theta_1.^2);

%grad_0 =  (1/m)*sum(X'(1,:)*(p-y));
%grad =  (1/m)*sum(X'(2,:)*(p-y))+(lambda/m)*theta_1;
%grad(1)= grad_0(1);
%============================












% =========================================================================

grad = grad(:);

end
