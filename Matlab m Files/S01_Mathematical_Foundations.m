%% Season 1: Mathematical Foundations for Modern Control
% *Prerequisites:* Basic linear algebra
% 
% *MATLAB Version:* R2025b
% 
% *Toolboxes Required:* Symbolic Math Toolbox

close all; clear; clc;
rng(0);  % For reproducibility
%% Section 0:
% Basic Matlab Functions:
% Basic arithmetic operators

Addition = 2 + 3
Subtraction = 2 - 3
Multiplication = 2 * 3
Division = 2 / 3
syms x y;

x = 2 + 3 * 4
x = (2 + 3) * 4
y = 1 / (2 + 3^2) + 4/5 * 6/7
y = 1 / 2 + 3^2 + 4/5 * 6/7
%% 
% 
% Mathematical functions

rad2deg(1)       % Radians to Degrees
deg2rad(90)      % Degrees to Radians
cosine_val = cos(x)  % Cosine 
sine_val = sin(x)    % Sine 
tan_val = tan(x)     % Tangent 
arcc_val = acos(x)   % Arc cosine
arcs_val = asin(x)   % Arc sine 
arct_val = atan(x)   % Arc tangent
exp_val = exp(x)     % Exponential 
sqrt_val = sqrt(x)   % Square root
log_val = log(x)     % Natural 
log10_val = log10(x) % Common logarithm
%% 
% 

x = -20;
abs_val = abs(x)     % Absolute value
sign_val = sign(x)   % Signum function
%% 
% 

x = [1 2 3 7 465 -2 7.6];
max_val = max(x)     % Maximum value
min_val = min(x)     % Minimum value
%% 
% 

x = 45.3;
ceil_val = ceil(x)   % Round towards +∞
floor_val = floor(x) % Round towards −∞
round_val = round(x) % Round to nearest integer
%% 
% 

x = 2 + 7i;
real(x)
imag(x)
angle_val = angle(x) % Phase angle
conj_val = conj(x)   % Complex conjugate
% Matrix

A = [1 2 3; 4 5 6; 7 8 9]
A(2,1)
A(3,3) = 0
m = 2;
n = 3;
eye1 = eye(m,n)    % Returns an m-by-n matrix with 1 on the main diagonal
eye2 = eye(n)      % Returns an n-by-n square identity matrix
zeros_example = zeros(m,n) % Returns an m-by-n matrix of zeros
ones_example = ones(m,n)   % Returns an m-by-n matrix of ones
diag_example = diag(A)     % Extracts the diagonal of matrix A
rand_example = rand(m,n)   % Returns an m-by-n matrix of random numbers
%% Section 1.1: Vector and Matrix Operations
% *Concept Overview*
% Matrix operations form the foundation of state-space control theory. Understanding 
% element-wise operations, matrix multiplication, and basic matrix properties 
% is essential for analyzing linear systems.

% Create sample matrices
A = [1 2 3; 4 5 6; 7 8 10];

fprintf('Matrix A:\n'); % Use fprintf to write data to the screen or a text file, refer to the documentation for writing to a text file
disp(A); % display the output without the variable name

B = [2 1 0; 1 3 1; 0 1 2]
disp(B);
v = [1; 2; 3];

% Basic matrix operations
% A + B (addition)

C_add = A + B
%% 
% A * B (matrix multiplication):

C_mult = A * B
%% 
% A .* B (element-wise multiplication):

C_elem = A .* B  
%% 
% A.^2 (Element-wise power)

C_power = A.^2  
%% 
% 
% 
% Transpose operations

A_transpose = A.'       % Non-conjugate transpose
A_hermitian = A'        % Conjugate transpose (same for real matrices)
%% 
% Matrix concatenation

H_concat = [A, B]       % Horizontal concatenation
V_concat = [A; B]       % Vertical concatenation
%% Section 1.2: Matrix Properties and Functions
% *Mathematical Background*
% 
% Key matrix properties:
%% 
% * Determinant: det(A) - nonzero for invertible matrices
% * Rank: number of linearly independent rows/columns
% * Trace: sum of diagonal elements, tr(A) = sum of eigenvalues
% * Inverse: A^(-1) exists if det(A) ≠ 0
%% 
% Determinant

det_A = det(A);
fprintf('Determinant of A: %.4f\n', det_A);
%% 
% Rank

rank_A = rank(A);
fprintf('Rank of A: %d (size %dx%d)\n', rank_A, size(A,1), size(A,2));
%% 
% Trace (_Sum of diagonal elements)_

trace_A = trace(A);
fprintf('Trace of A: %.4f\n', trace_A);
%% 
% Inverse (if exists)

if det_A ~= 0
    A_inv = inv(A);
    fprintf('Inverse of A:\n');
    disp(A_inv);
    
    % Verify A * inv(A) = I
    identity_check = A * A_inv;
    fprintf('A * inv(A) (should be identity):\n');
    disp(identity_check);
else
    fprintf('Matrix A is singular (not invertible)\n');
end
% Condition number (measures sensitivity to perturbations)
% A _condition number_ for a matrix and computational task measures how sensitive 
% the answer is to changes in the input data and roundoff errors in the solution 
% process.
% 
% The _condition number for inversion_ of a matrix measures the sensitivity 
% of the solution of a system of linear equations to errors in the data. It gives 
% an indication of the accuracy of the results from matrix inversion and the linear 
% equation solution. For example, the 2-norm condition number of a square matrix 
% is
% 
% $$\kappa (A)=\|A\|\|A-1\|\,.$$
% 
% In this context, a large condition number indicates that a small change in 
% the coefficient matrix |A| can lead to larger changes in the output |b| in the 
% linear equations _Ax_ = _b_ and _xA_ = _b_. The extreme case is when |A| is 
% so poorly conditioned that it is singular (an infinite condition number), in 
% which case it has no inverse and the linear equation has no unique solution.

cond_A = cond(A);
fprintf('Condition number of A: %.4f\n', cond_A);
fprintf('(Higher values indicate ill-conditioning)\n\n');
%% Section 1.3: Matrix Norms
% *Concept Overview*
% 
% Matrix norms measure the "size" of a matrix and are crucial for:
%% 
% * Stability analysis
% * Error bounds in numerical computations
% * Convergence analysis
%% 
% *Common Norms:*
%% 
% * 1-norm: maximum absolute column sum
% * 2-norm (spectral): largest singular value
% * ∞-norm: maximum absolute row sum
% * Frobenius norm: sqrt(sum of squared elements)
%% 
% Create a test matrix with complex entries

A_complex = [1-2i, 2-1i, 5; 7, 5+4i, 3; 3, 8, 9+1i];
%% 
% Different norm types
% 
% 1-norm (max column sum)

norm_1 = norm(A_complex, 1)
%% 
% 2-norm (spectral/largest singular value)

norm_2 = norm(A_complex, 2)
%% 
% inf-norm (max row sum)

norm_inf = norm(A_complex, inf)
%% 
% Frobenius norm

norm_fro = norm(A_complex, 'fro')
%% 
% Vector norms

v_test = [3; 4; 0];
v_norm1 = norm(v_test, 1);      % Sum of absolute values
v_norm2 = norm(v_test, 2);      % Euclidean norm
v_norminf = norm(v_test, inf);  % Maximum absolute value
disp(v_test);
fprintf('1-norm: %.4f\n', v_norm1);
fprintf('2-norm (Euclidean): %.4f\n', v_norm2);
fprintf('inf-norm: %.4f\n\n', v_norminf);
%% Section 1.4: Triangular Matrices
% *Concept Overview*
% 
% Upper and lower triangular matrices are important for:
%% 
% * Efficient solving of linear systems
% * QR and LU decompositions
% * Analyzing stability (eigenvalues on diagonal)

M = [3.1, -1.6, 11.1; -8.6, 6.2, -8; -0.3, 11, 0.7]
% Extract upper and lower triangular parts
% Upper triangular

U = triu(M)  % Upper triangular
%% 
% Lower triangular

L = tril(M)  % Lower triangular
%% 
% Diagonal extraction

D = diag(M) % Extract diagonal elements
%% Section 1.5: Linear Equations, Null Space, and Orthogonality
% *Mathematical Background*
% 
% For system Ax = b:
%% 
% * Null space N(A): all vectors x where Ax = 0
% * Range/Column space R(A): all possible outputs Ax
% * Orthogonal complement: vectors perpendicular to a subspace
%% 
% *Key Property:* rank(A) + dim(null(A)) = n (number of columns)
% 
% Create a rank-deficient matrix

A_rank_def = [1 2 3; 2 3 4; 4 5 6; 25 34.5 44];

fprintf('Rank-deficient matrix A:\n');
disp(A_rank_def);

fprintf('Rank: %d, Columns: %d\n', rank(A_rank_def), size(A_rank_def, 2));
% Null space
% Null space basis (should satisfy A*Z ≈ 0):

Z = null(A_rank_def)
%% 
% Verify (A * null(A) should be ≈ 0)

verification = A_rank_def * Z
fprintf('Norm of A*null(A): %.2e\n\n', norm(verification));
%% 
% Orthonormal basis for column space

Q = orth(A_rank_def)
%% 
% Verify orthonormality: Q'*Q should be identity

orthogonality_check = Q' * Q;
fprintf('Q^T * Q (should be identity of size %dx%d):\n', size(Q,2), size(Q,2));
disp(orthogonality_check);
%% 
% Dot product and orthogonality check

u = randi(10, [3,1]); %what's the difference between randi and rand?
v = randi(10, [3,1]);
dot_uv = dot(u, v);
fprintf('\nVectors u and v:\n');
fprintf('u = '); disp(u');
fprintf('v = '); disp(v');
fprintf('Dot product u·v = %.4f\n', dot_uv);
if abs(dot_uv) < 1e-10
    fprintf('Vectors are orthogonal\n');
else
    fprintf('Vectors are not orthogonal\n');
end
%% Solving linear equations Ax = b

A_solve = [2 1 -1; -3 -1 2; -2 1 2];
b_solve = [8; -11; -3];
%% 
% Using backslash operator (most efficient)

x_solution = A_solve \ b_solve;

fprintf('\nSolving Ax = b:\n');
fprintf('A:\n'); disp(A_solve);
fprintf('b:\n'); disp(b_solve');
fprintf('Solution x:\n'); disp(x_solution');
%% 
% Verify solution

residual = norm(A_solve * x_solution - b_solve);
fprintf('Residual ||Ax - b||: %.2e\n', residual);
% Alternative: using linsolve for more control

[x_linsolve, R] = linsolve(A_solve, b_solve);
fprintf('Linsolve solution is:\n'); disp(x_linsolve);
fprintf('Reciprocal condition estimate: %.2e\n', R);
%% Section 1.6: Eigenvalues and Eigenvectors
% *Mathematical Background*
% 
% For matrix A, eigenvalue λ and eigenvector v satisfy:
% 
% $$Av = \lambda v$$
% 
% *Key Properties:*
%% 
% * Characteristic polynomial: det(A - λI) = 0
% * Trace = sum of eigenvalues
% * Determinant = product of eigenvalues
% * Eigenvalues determine stability of dynamic systems

A_eig = [5 11 4; 12 8 5; 1 7 3];

fprintf('Matrix A:\n'); disp(A_eig);
%% 
% Compute eigenvalues and eigenvectors

[V, D] = eig(A_eig);

fprintf('Eigenvalues (diagonal of D):\n'); disp(diag(D));

fprintf('Eigenvector matrix V:\n'); disp(V);
%% 
% Verify: A*V should equal V*D

verification_eig = A_eig * V
expected_eig = V * D

fprintf('Verification: max|A*V - V*D| = %.2e\n', max(max(abs(verification_eig - expected_eig))));
%% 
% Characteristic polynomial (using poly)

char_poly = poly(A_eig);
fprintf('\nCharacteristic polynomial coefficients:\n');
fprintf('p(λ) = ');
for i = 1:length(char_poly)
    if i == 1
        fprintf('%.4fλ^%d', char_poly(i), length(char_poly)-i);
    else
        if char_poly(i) >= 0
            fprintf(' + %.4fλ^%d', char_poly(i), length(char_poly)-i);
        else
            fprintf(' - %.4fλ^%d', abs(char_poly(i)), length(char_poly)-i);
        end
    end
end
% Alternative: use charpoly

syms lambda;
charpoly(A_eig,lambda)
%% 
% Properties

trace_sum = sum(diag(D));
det_prod = prod(diag(D)); % product of the array elements

fprintf('Trace of A: %.4f\n', trace(A_eig));
fprintf('Sum of eigenvalues: %.4f\n', trace_sum);
fprintf('Determinant of A: %.4f\n', det(A_eig));
fprintf('Product of eigenvalues: %.4f\n\n', det_prod);
%% Section 1.7: Similarity Transformations
% *Concept Overview*
% 
% Two matrices A and B are similar if: B = T^(-1) * A * T
% 
% Similar matrices have:
%% 
% * Same eigenvalues
% * Same determinant, trace, rank
% * Same characteristic polynomial
%% 
% *Application:* Transform systems to diagonal or canonical forms
% 
% Using eigenvector matrix as transformation

A_original = A_eig;
T = V;  % Eigenvector matrix

if abs(det(T)) > 1e-10
    A_transformed = inv(T) * A_original * T;
    
    fprintf('Original matrix A:\n');
    disp(A_original);
    
    fprintf('Transformation matrix T (eigenvectors):\n');
    disp(T);
    
    fprintf('Transformed matrix T^(-1)*A*T (should be diagonal):\n');
    disp(A_transformed);
    
    % Check eigenvalues are preserved
    eig_original = sort(eig(A_original));
    eig_transformed = sort(eig(A_transformed));
    
    fprintf('Eigenvalues of original: ');
    fprintf('%.4f ', eig_original);
    fprintf('\n');
    fprintf('Eigenvalues of transformed: ');
    fprintf('%.4f ', eig_transformed);
    fprintf('\n\n');
end
%% Section 1.8: Jordan Normal Form
% *Mathematical Background*
% 
% Every square matrix is similar to its Jordan normal form:
% 
% $$A = T J T^{-1}$$
% 
% where J is block-diagonal with Jordan blocks.
% 
% *Jordan Block:* Upper triangular matrix with eigenvalue on diagonal
% 
% Matrix with repeated eigenvalues

A_jordan = [5 11 4; 12 8 5; 1 7 3];

fprintf('Matrix A:\n');
disp(A_jordan);

%% 
% Compute Jordan form

[T_jordan, J] = jordan(A_jordan);

fprintf('Jordan form J:\n');
disp(J);

fprintf('Transformation matrix T:\n');
disp(T_jordan);

%% 
% Verify: A = T*J*inv(T)

A_reconstructed = T_jordan * J * inv(T_jordan);
fprintf('Reconstructed A from T*J*T^(-1):\n');
disp(real(A_reconstructed));

reconstruction_error = norm(A_jordan - A_reconstructed);
fprintf('Reconstruction error: %.2e\n\n', reconstruction_error);

%% 
% Example with defective matrix (non-diagonalizable)

A_defective = [2 1 0; 0 2 0; 0 0 3];
fprintf('Defective matrix (non-diagonalizable):\n');
disp(A_defective);

[T_def, J_def] = jordan(A_defective);
fprintf('Jordan form (note the 1 above diagonal for repeated eigenvalue):\n');
disp(J_def);
%% Section 1.9: Matrix Exponential
% *Mathematical Background*
% 
% Matrix exponential is defined as:
% 
% $$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + ...$$
% 
% *Key Application:* Solution to linear differential equations
% 
% For $\dot{x} = Ax$, solution is $x(t) = e^{At}x_0$
% 
% State matrix for a stable system

A_exp = [0 1; -2 -3];

fprintf('Matrix A (typical state matrix):\n');
disp(A_exp);
%% 
% Compute matrix exponential at t = 0

exp_A0 = expm(A_exp * 0);
fprintf('exp(A*0) (should be identity):\n');
disp(exp_A0);
%% 
% Compute at t = 1

t = 1;
exp_At = expm(A_exp * t);
fprintf('exp(A*%.1d):\n', t);
disp(exp_At);
%% 
% State transition: if x(0) = [1; 0], what is x(1)?

x0 = [1; 0];
x_t = exp_At * x0;
fprintf('If x(0) = [1; 0], then x(%.1d) = exp(A*%.1d)*x(0) =\n', t, t);
disp(x_t');
%% 
% Visualize state trajectory

t_span = linspace(0, 5, 100); %from 0 to 5, 100 points inbetween
x_trajectory = zeros(2, length(t_span));

for i = 1:length(t_span)
    x_trajectory(:,i) = expm(A_exp * t_span(i)) * x0;
end

figure('Name', 'State Trajectory using Matrix Exponential');
subplot(2,1,1);
plot(t_span, x_trajectory(1,:), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('x_1(t)');
title('State x_1 vs Time');

subplot(2,1,2);
plot(t_span, x_trajectory(2,:), 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('x_2(t)');
title('State x_2 vs Time');
%% 
% Phase portrait

figure('Name', 'Phase Portrait');
plot(x_trajectory(1,:), x_trajectory(2,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(x0(1), x0(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(x_trajectory(1,end), x_trajectory(2,end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
grid on;
xlabel('x_1');
ylabel('x_2');
title('Phase Portrait: x_2 vs x_1');
legend('Trajectory', 'Initial', 'Final', 'Location', 'best');
axis equal;

fprintf('\nSystem decays to zero (stable) as eigenvalues have negative real parts\n');
fprintf('Eigenvalues of A: ');
fprintf('%.4f ', eig(A_exp));
fprintf('\n');
%% Section 1.10: Summary and Key Takeaways
% *Key Concepts Covered:*
%% 
% # Matrix operations: addition, multiplication, transpose, concatenation
% # Matrix properties: determinant, rank, trace, inverse, condition number
% # Matrix norms: 1-norm, 2-norm, infinity-norm, Frobenius norm
% # Triangular matrices: upper, lower, diagonal
% # Linear equations: solving Ax=b, null space, orthogonality
% # Eigenanalysis: eigenvalues, eigenvectors, characteristic polynomial
% # Similarity transformations and invariants
% # Jordan normal form for matrices with repeated eigenvalues
% # Matrix exponential for solving linear differential equations
%% 
% *MATLAB Functions Mastered:*
% 
% |eig|, |inv|, |det|, |rank|, |trace|, |norm|, |triu|, |tril|, |null|, |orth|, 
% |dot|, |linsolve|, |poly|, |jordan|, |expm|
% 
% *Next Steps:*
% 
% These mathematical tools will be applied to:
%% 
% * State-space modeling (Season 2)
% * Controllability and observability analysis (Season 3)
% * Stability analysis using Lyapunov theory (Season 5)
% * Observer and controller design (Season 6)