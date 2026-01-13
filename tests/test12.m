% Test 12: Incompatible reassignment (warning expected)

A = zeros(3, 4);
A = zeros(5, 4);   % incompatible reassignment: 3x4 -> 5x4