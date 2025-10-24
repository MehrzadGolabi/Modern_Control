%% Season 3 Exercises: Controllability, Observability, and Realization
close all; clear; clc;

% Enter your FULL student number below (e.g., 400249049)
STUDENT_NUMBER = 0;  % REPLACE THIS WITH YOUR STUDENT NUMBER
rng(STUDENT_NUMBER);

%%% Name :
%%% Student number =

%% Exercise 3.1: Controllability - Rank Test
% 1. *Problem 3.1.1:*
% Test controllability using rank criterion:
% A = [0 1; -2 -3], B = [0; 1]
% (a) Compute controllability matrix C = [B AB]
% (b) Find rank of controllability matrix
% (c) Is the system controllable?
% (d) Verify using ctrb() function

% Your code here:


%% 
% 2. *Problem 3.1.2:*
% For the system: A = [1 2; 0 3], B = [1; 2]
% (a) Compute controllability matrix manually
% (b) Check rank - is it controllable?
% (c) If not controllable, which states are uncontrollable?
% (d) Verify your result using ctrb()

% Your code here:


%% 
% 3. *Problem 3.1.3:*
% Third-order system: A = [0 1 0; 0 0 1; -6 -11 -6], B = [0; 0; 1]
% (a) Compute C = [B AB A²B]
% (b) Find rank of C
% (c) Is the system controllable?
% (d) What is the dimension of the controllable subspace?

% Your code here:


%% Exercise 3.2: Observability - Rank Test
% 4. *Problem 3.2.1:*
% Test observability:
% A = [0 1; -2 -3], C = [1 0]
% (a) Compute observability matrix O = [C; CA]
% (b) Find rank of O
% (c) Is the system observable?
% (d) Verify using obsv() function

% Your code here:


%% 
% 5. *Problem 3.2.2:*
% For A = [-1 1; -2 0], C = [1 1]
% (a) Compute observability matrix
% (b) Check rank - is it observable?
% (c) What does it mean physically if unobservable?
% (d) Try different C = [1 0] - does observability change?

% Your code here:


%% 
% 6. *Problem 3.2.3:*
% Multi-output system:
% A = [0 1 0; 0 0 1; -1 -2 -3], C = [1 0 0; 0 1 0]
% (a) Compute observability matrix O
% (b) What is the size of O?
% (c) Find rank and determine observability
% (d) Compare with single output C = [1 0 0]

% Your code here:


%% Exercise 3.3: PBH (Popov-Belevitch-Hautus) Test
% 7. *Problem 3.3.1:*
% PBH Controllability Test:
% A = [1 1; 0 2], B = [1; 1]
% (a) Find eigenvalues of A
% (b) For each eigenvalue λ, form [λI-A, B]
% (c) Check rank for each eigenvalue
% (d) Is the system controllable by PBH test?
% (e) Verify with rank test using ctrb()

% Your code here:


%% 
% 8. *Problem 3.3.2:*
% PBH Observability Test:
% A = [0 1; -4 -5], C = [1 1]
% (a) Find eigenvalues of A
% (b) For each eigenvalue λ, form [λI-A; C]
% (c) Check rank for each eigenvalue
% (d) Is the system observable by PBH test?

% Your code here:


%% Exercise 3.4: Controllability and Observability Gramians
% 9. *Problem 3.4.1:*
% For A = [0 1; -1 -2], B = [0; 1], C = [1 0]
% (a) Compute controllability gramian using gram()
% (b) Find eigenvalues of Wc
% (c) Is Wc positive definite?
% (d) Conclude about controllability

% Your code here:


%% 
% 10. *Problem 3.4.2:*
% For the same system:
% (a) Compute observability gramian using gram()
% (b) Find eigenvalues of Wo
% (c) Is Wo positive definite?
% (d) Conclude about observability

% Your code here:


%% 
% 11. *Problem 3.4.3:*
% Create an uncontrollable system:
% A = [1 0; 0 2], B = [1; 0]
% (a) Verify it's uncontrollable using rank test
% (b) Try to compute gramian - what happens?
% (c) If gramian can be computed, check if positive definite
% (d) Explain the relationship between rank and gramian tests

% Your code here:


%% Exercise 3.5: Kalman Decomposition and Minimal Realization
% 12. *Problem 3.5.1:*
% System with uncontrollable modes:
% A = [-1 0 0; 0 -2 0; 0 0 -3], B = [1; 1; 0], C = [1 1 1]
% (a) Check controllability using ctrb()
% (b) Check observability using obsv()
% (c) Use minreal() to find minimal realization
% (d) How many states were removed?

% Your code here:


%% 
% 13. *Problem 3.5.2:*
% Compare full and minimal realizations:
% A = [0 1 0 0; -1 -1 0 0; 0 0 -2 1; 0 0 -3 -2]
% B = [0; 1; 0; 0], C = [1 0 0 0]
% (a) Create state-space system
% (b) Find minimal realization
% (c) Compare transfer functions of both
% (d) Plot step responses - are they identical?

% Your code here:


%% 
% 14. *Problem 3.5.3:*
% Pole-zero cancellation:
% Create H(s) = (s+1)(s+2)/[(s+1)(s+3)(s+4)]
% (a) Create transfer function
% (b) Convert to state-space using ss()
% (c) Apply minreal()
% (d) Verify the cancelled pole is removed

% Your code here:


%% Exercise 3.6: Canonical Forms
% 15. *Problem 3.6.1:*
% Controllable Canonical Form:
% A = [0 1; -6 -5], B = [0; 1], C = [1 0]
% (a) Transform to controllable canonical form using canon()
% (b) Display A_c and B_c matrices
% (c) Verify B_c has standard form [0; 0; ...; 1]
% (d) Extract characteristic polynomial from A_c

% Your code here:


%% 
% 16. *Problem 3.6.2:*
% Observable Canonical Form:
% For same system as 3.6.1
% (a) Use duality: create dual system
% (b) Transform dual to controllable canonical
% (c) Transform back to get observable canonical
% (d) Verify the transformation

% Your code here:


%% 
% 17. *Problem 3.6.3:*
% Modal (Diagonal) Canonical Form:
% A = [0 1; -8 -6], B = [0; 1], C = [1 0]
% (a) Transform to modal form using canon()
% (b) Verify A_modal is diagonal
% (c) Check that diagonal elements are eigenvalues of A
% (d) What is physical interpretation of modal form?

% Your code here:


%% Exercise 3.7: State-Space Transformations
% 18. *Problem 3.7.1:*
% Given: A = [0 1; -2 -3], B = [0; 1], C = [1 1], D = 0
% Transformation: T = [1 1; 0 1]
% (a) Compute transformed system manually: Ā = TAT⁻¹, B̄ = TB, C̄ = CT⁻¹
% (b) Verify using ss2ss() function
% (c) Compare transfer functions before and after
% (d) Are eigenvalues preserved?

% Your code here:


%% 
% 19. *Problem 3.7.2:*
% Diagonalization using eigenvectors:
% A = [1 2; 3 4], B = [1; 0], C = [1 0]
% (a) Find eigenvalues and eigenvectors of A
% (b) Form transformation matrix T from eigenvectors
% (c) Transform system to modal form
% (d) Verify A_modal is diagonal with eigenvalues

% Your code here:


%% Exercise 3.8: Balanced Realization
% 20. *Problem 3.8.1:*
% System: A = [-1 0 0.5; 0 -2 0; 0 0 -5], B = [1; 0.5; 0.1], C = [1 1 0.5]
% (a) Compute controllability and observability gramians
% (b) Transform to balanced realization using balreal()
% (c) Display Hankel singular values
% (d) Which states are most important?

% Your code here:


%% 
% 21. *Problem 3.8.2:*
% Model reduction using balanced truncation:
% For the system in 3.8.1
% (a) Identify states with small Hankel singular values
% (b) Remove least important states (keep σ > 0.1*σ_max)
% (c) Create reduced-order model
% (d) Compare step responses of full and reduced models
% (e) Calculate approximation error

% Your code here:


%% 
% 22. *Problem 3.8.3:*
% Fourth-order system:
% A = [-1 0 0.5 0; 0 -2 0 0.3; 0 0 -5 0; 0 0 0 -10]
% B = [1; 0.5; 0.2; 0.1], C = [1 1 0.5 0.1]
% (a) Compute Hankel singular values
% (b) Plot singular values to visualize state importance
% (c) Reduce to 2-state model
% (d) Compare frequency responses (use bode plot)

% Your code here:


%% Exercise 3.9: Combined Controllability and Observability
% 23. *Problem 3.9.1:*
% Analyze all four structural properties:
% A = [0 1; -2 -1], B = [0; 1], C = [1 0]
% (a) Test controllability (rank method)
% (b) Test observability (rank method)
% (c) Verify with PBH tests
% (d) Verify with gramian tests
% (e) Summarize results in a table

% Your code here:


%% 
% 24. *Problem 3.9.2:*
% System that is controllable but not observable:
% A = [-1 0; 0 -2], B = [1; 1], C = [1 1]
% (a) Verify controllability
% (b) Check observability - explain why it fails
% (c) Find minimal realization
% (d) Can you design C to make it observable?

% Your code here:


%% 
% 25. *Problem 3.9.3:*
% System that is observable but not controllable:
% A = [0 1; 0 -1], B = [0; 0], C = [1 0]
% (a) Check controllability - why does it fail?
% (b) Verify observability
% (c) Can this system be controlled by any input?
% (d) What is the physical interpretation?

% Your code here:


%% Exercise 3.10: Practical Applications
% 26. *Problem 3.10.1:*
% Mass-spring-damper with partial state measurement:
% A = [0 1; -4 -2], B = [0; 1], C = [1 0] (measure position only)
% (a) Is the system controllable?
% (b) Is it observable from position measurement only?
% (c) Try C = [0 1] (measure velocity only) - is it observable?
% (d) Try C = [1 1] (measure position + velocity) - how about now?

% Your code here:


%% 
% 27. *Problem 3.10.2:*
% DC Motor - analyze structural properties:
% A = [-R/L -Kb/L; Kt/J -B/J], B = [1/L; 0], C = [0 1]
% Use R=1, L=0.5, Kt=0.01, Kb=0.01, J=0.01, B=0.1
% (a) Check controllability from voltage input
% (b) Check observability from velocity measurement
% (c) Compute gramians
% (d) Is this a minimal realization?

% Your code here:


%% 
% 28. *Problem 3.10.3:*
% Satellite attitude control:
% States: [θ, θ̇, ψ, ψ̇] (two axes)
% A = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
% B = [0; 1; 0; 0], C = [1 0 0 0] (measure θ only)
% (a) Check controllability - can we control both axes?
% (b) Check observability - can we observe all states?
% (c) Suggest improvements to B or C for full controllability/observability

% Your code here:


%% Exercise 3.11: Comprehensive Problems
% 29. *Problem 3.11.1:*
% Complete structural analysis:
% A = [0 1 0; 0 0 1; -1 -2 -1], B = [0; 0; 1], C = [1 0 0]
% Perform complete analysis:
% (a) Controllability: rank test, PBH test, gramian
% (b) Observability: rank test, PBH test, gramian
% (c) Transform to all three canonical forms
% (d) Compute balanced realization
% (e) Compare eigenvalues across all forms

% Your code here:


%% 
% 30. *Problem 3.11.2:*
% Model reduction workflow:
% Start with 5th-order system:
% A = diag([-1, -2, -3, -10, -20]), B = ones(5,1), C = ones(1,5)
% (a) Check controllability and observability
% (b) Compute balanced realization
% (c) Plot Hankel singular values
% (d) Reduce to 2-state model
% (e) Quantify approximation error in step response

% Your code here:


%% 
% 31. *Problem 3.11.3:*
% Transfer function realization comparison:
% H(s) = (s+2)/(s³+6s²+11s+6)
% (a) Find poles and zeros
% (b) Convert to state-space (default realization)
% (c) Transform to controllable canonical form
% (d) Transform to observable canonical form
% (e) Transform to modal canonical form
% (f) Verify all have same transfer function

% Your code here:


%% 
% 32. *Problem 3.11.4:*
% Design for controllability and observability:
% Given A = [0 1; -4 -4], design B and C such that:
% (a) System is controllable: try different B vectors
% (b) System is observable: try different C vectors
% (c) Find a B that makes it uncontrollable
% (d) Find a C that makes it unobservable
% (e) Explain the geometric interpretation

% Your code here:



