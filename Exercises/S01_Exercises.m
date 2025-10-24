%% Season 1 Exercises: Mathematical Foundations for Modern Control
close all; clear; clc;

% Enter your FULL student number below (e.g., 400249123)
STUDENT_NUMBER = 0;  % REPLACE THIS WITH YOUR STUDENT NUMBER
rng(STUDENT_NUMBER);

% Note: Your values will differ based on your STUDENT_NUMBER

%%% Name :
%%% Student number =

%% Exercise 1.1: Basic Matrix Operations
% 1. *Problem 1.1.1:* 
% Create two 3×3 matrices A and B with random integers (using randi) between 1 and 10. Compute:

% (a) A + B
% (b) A x B (matrix multiplication)
% (c) A x B (element-wise multiplication)
% (d) A^2 and A.^2 - explain the difference

% Your code here:


%% 
% 2. *Problem 1.1.2:*
% Given vectors u = [1; 2; 3] and v = [4; 5; 6], compute:
% (a) dot product as matrix multiplication
% (b) outer product
% (c) The angle between u and v (hint: use dot product formula)

% Your code here:


%% 
% 3. *Problem 1.1.3:* 
% Create a 4×4 magic square using the magic() function.
% Extract:
% (a) The 2×2 submatrix in the top-left corner
% (b) The second row
% (c) The third column
% (d) The main diagonal and anti-diagonal

% Your code here:


%% Exercise 1.2: Matrix Properties
% 4. *Problem 1.2.1:* 
% Create a random 5×5 matrix A. 
% Compute and verify:
% (a) determinant
% (b) rank
% (c) trace
% (d) Verify that trace of A = sum of eigenvalues
% (e) Verify that determinant of A = product of eigenvalues

% Your code here:


%% 
% 5. *Problem 1.2.2:* Create a singular (non-invertible) 3×3 matrix.
% (a) Verify that its determinant is zero
% (b) Show that it has at least one zero eigenvalue
% (c) Find its null space dimension

% Your code here:


%% Exercise 1.3: Matrix Norms
% 6. *Problem 1.3.1:*
% For matrix A = [1 -2 3; 4 5 -6; -7 8 9]:
% (a) Compute all four norms: 1-norm, 2-norm, inf-norm, Frobenius norm
% (b) Manually verify the 1-norm (maximum absolute column sum)
% (c) Manually verify the inf-norm (maximum absolute row sum)

% Your code here:


%% 
% 7. *Problem 1.3.2:*
% Create two random 3×3 matrices A and B.
% (a) Verify the triangle inequality: ||A + B||_F ≤ ||A||_F + ||B||_F
% (b) Verify submultiplicativity: ||A*B||_2 ≤ ||A||_2 * ||B||_2

% Your code here:


%% Exercise 1.4: Triangular Matrices
% 8. *Problem 1.4.1:* 
% Create a random 4×4 matrix M.
% (a) Extract upper triangular part U
% (b) Extract lower triangular part L
% (c) Extract diagonal D
% (d) Verify that M = U + L - diagonal(diagonal(M))

% Your code here:


%% 
% 9. *Problem 1.4.2:*
% Solve the upper triangular system Ux = b where:
% U = [4 3 2 1; 0 3 2 1; 0 0 2 1; 0 0 0 1]
% b = [10; 6; 3; 1]

% (a) Solve using backslash operator
% (b) Solve manually using back-substitution (starting from last row)
% (c) Verify both solutions are identical

% Your code here:


%% Exercise 1.5: Null Space and Orthogonality
% 10. *Problem 1.5.1:*
% For matrix A = [1 2 3; 2 4 6; 1 2 3]:
% (a) Find the rank of A
% (b) Find a basis for the null space
% (c) Verify that A * null(A) ≈ 0
% (d) Verify: rank(A) + dim(null(A)) = number of columns

% Your code here:


%% 
% 11. *Problem 1.5.2:*
% Create an orthonormal basis for the column space of:
% A = [1 2; 2 1; 2 2; 1 0]
% (a) Use orth() function
% (b) Verify that Q'*Q = I where Q = orth(A)
% (c) Verify that columns of Q are unit vectors

% Your code here:


%% 
% 12. *Problem 1.5.3:*
% Gram-Schmidt Orthogonalization
% Given vectors: v1 = [1; 1; 0], v2 = [1; 0; 1], v3 = [0; 1; 1]
% (a) Apply Gram-Schmidt process to create orthonormal basis
% (b) Verify orthogonality using dot products
% (c) Compare with result from orth([v1 v2 v3])

% Your code here:


%% Exercise 1.6: Solving Linear Systems
% 13. *Problem 1.6.1:* 
% Solve the system Ax = b where:
% A = [2 -1 1; 1 1 -1; 1 -1 2]
% b = [6; 0; 3]
% (a) Solve using x = A\b
% (b) Solve using x = inv(A)*b
% (c) Compare computation time for both methods (use tic/toc or timeit)
% (d) Which method is more efficient?

% Your code here:


%% Exercise 1.7: Eigenvalues and Eigenvectors
% 14. *Problem 1.7.1:* For matrix A = [4 1; 2 3]:
% (a) Compute eigenvalues and eigenvectors using [V,D] = eig(A)
% (b) Verify that A*V = V*D
% (c) For each eigenvector v_i and eigenvalue λ_i, verify A*v_i = λ_i*v_i
% (d) Compute eigenvalues manually using characteristic equation det(A - λI) = 0

% Your code here:


%% 
% 15. *Problem 1.7.2:* 
% Investigate the relationship between matrix properties:
% Create A = [6 -1 0; -1 5 -1; 0 -1 4]
% (a) Find characteristic polynomial using poly(A)
% (b) Verify polynomial roots equal eigenvalues

% Your code here:


%% Exercise 1.8: Similarity Transformations
% 16. *Problem 1.8.1:* 
% For matrix A = [1 2; 3 4], transformation T = [2 1; 1 1]:
% (a) Compute B = inv(T)*A*T
% (b) Verify that A and B have the same eigenvalues
% (c) Verify that A and B have the same determinant
% (d) Verify that A and B have the same trace

% Your code here:


%% 
% 17. *Problem 1.8.2:* 
% Diagonalization
% For matrix A = [5 4; 1 2]:
% (a) Find eigenvalues and eigenvectors
% (b) Form diagonal matrix D and transformation matrix T
% (c) Verify that A = T*D*inv(T)
% (d) Compute A^10 using diagonalization: A^10 = T*D^10*inv(T)
% (e) Compare with direct computation of A^10

% Your code here:


%% Exercise 1.9: Jordan Normal Form
% 18. *Problem 1.9.1:* 
% For matrix A = [3 1 0; 0 3 1; 0 0 3]:
% (a) Find eigenvalues - what do you notice?
% (b) Compute Jordan form using [T,J] = jordan(A)
% (c) Verify A = T*J*inv(T)
% (d) Try to diagonalize A - what happens?

% Your code here:


%% 
% 19. *Problem 1.9.2:* 
% For matrix A = [2 1 0 0; 0 2 0 0; 0 0 3 1; 0 0 0 3]:
% (a) Compute Jordan form
% (b) Identify the Jordan blocks

% Your code here:


%% Exercise 1.10: Matrix Exponential
% 20. *Problem 1.10.1:* 
% For A = [0 1; -1 0] (represents harmonic oscillator):
% (a) Compute exp(A*t) for t = 0, π/2, π, 2π
% (b) What pattern do you observe?
% (c) Solve dx/dt = Ax with x(0) = [1; 0] for t ∈ [0, 4π]
% (d) Plot phase portrait

% Your code here:


%% 
% 21. *Problem 1.10.2:* 
% For diagonal matrix D = diag([λ1, λ2, λ3]):
% (a) Create D with λ = [-1, -2, -3]
% (b) Compute exp(D*t) for various t values
% (c) Verify that exp(D*t) = diag([exp(λ1*t), exp(λ2*t), exp(λ3*t)])
% (d) What happens as t → ∞?

% Your code here:


%% 
% 22. *Problem 1.10.3:* Using matrix exponential to solve initial value problem:
% For A = [-1 2; -2 -1], x(0) = [1; 1]:
% (a) Compute state transition matrix Φ(t) = exp(A*t)
% (b) Solution is x(t) = Φ(t)*x(0), compute for t = 0:0.1:5
% (c) Plot both states vs time
% (d) Create phase portrait
% (e) Determine if system is stable by examining eigenvalues

% Your code here:


%% Exercise 1.11: Comprehensive Problems
% 23. *Problem 1.11.1:* 
% Matrix Analysis Challenge
% Given A = [4 2 1; 0 3 2; 0 0 2]:
% Perform complete analysis:
% (a) All basic properties (det, rank, trace, condition number)
% (b) All four matrix norms
% (c) Eigenvalues and eigenvectors
% (d) Jordan form
% (e) Matrix exponential at t = 1
% (f) Solve Ax = b for b = [14; 11; 4]

% Your code here:


%% 
% 24. *Problem 1.11.4:* 
% Numerical Stability Investigation
% Create a Hilbert matrix: H = hilb(n) for n = 5, 10, 15
% (a) Compute condition numbers for each size
% (b) Solve H*x = b where b = H*ones(n,1) (so true solution is x = ones(n,1))
% (c) Compute error: norm(x - ones(n,1))
% (d) How does error grow with n?
% (e) Explain relationship between condition number and solution accuracy

% Your code here:


%% 
% 25. *Problem 1.11.5:* 
% Matrix Functions Comparison
% For A = [1 2; 3 4]:
% (a) Compute exp(A) using expm()
% (b) Compute exp(A) using diagonalization: exp(A) = T*exp(D)*inv(T)
% (c) Compute exp(A) using Taylor series (first 20 terms)
% (d) Compare all three methods - do they agree?
% (e) Which method is most accurate? Most efficient? (calculate the computation time hint: use tic/toc or timeit)

% Your code here:

