% Test 6: Valid symbolic dimension multiplication, no warnings are expected
% A is m x n and B is n x m, so A * B is m x m and valid

m = 3;
n = 4;
A = zeros(m, n);
B = zeros(n, m);
C = A * B;