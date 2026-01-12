% Test 5: Valid colon vector and transpose, no warnings are expected
% v = 1:n is a 1xn row vector, so v' is nx1
% A * v' is valid when A is nxn

n = 5;
v = 1:n;
A = ones(n, n);
y = A * v';