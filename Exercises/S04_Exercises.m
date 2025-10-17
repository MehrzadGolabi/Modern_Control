%% Season 4 Exercises: Stability Analysis, State Feedback, and LQR
close all; clear; clc;

% Put the last 2 digits of your student number in the RNG function:
rng(0);

%%% Name :
%%% Student number =

%% Exercise 4.1: Stability Analysis - Eigenvalue Method
% *Problem 4.1.1:*
% Analyze stability using eigenvalues:
% A = [-2 1; 0 -3]
% (a) Compute eigenvalues of A
% (b) Plot eigenvalues on complex plane
% (c) Determine if system is stable, marginally stable, or unstable
% (d) Verify by simulating system response with initial condition

% Your code here:


%% 
% *Problem 4.1.2:*
% For oscillatory system: A = [0 2; -2 0]
% (a) Find eigenvalues
% (b) Classify stability (stable/marginally stable/unstable)
% (c) Plot phase portrait for different initial conditions
% (d) What type of behavior does this system exhibit?

% Your code here:


%% 
% *Problem 4.1.3:*
% Third-order system: A = [-1 0 0; 0 0 1; 0 -4 -5]
% (a) Compute all eigenvalues
% (b) Check stability
% (c) Identify modes: real vs complex, stable vs unstable
% (d) Plot eigenvalues showing stability boundaries

% Your code here:


%% Exercise 4.2: Lyapunov Stability Analysis
% *Problem 4.2.1:*
% Test stability using Lyapunov equation:
% A = [-1 0.5; -0.5 -2]
% (a) Solve Lyapunov equation A^TP + PA + Q = 0 with Q = I
% (b) Check if P is positive definite (compute eigenvalues)
% (c) Verify the solution: compute residual A^TP + PA + Q
% (d) Conclude about stability

% Your code here:


%% 
% *Problem 4.2.2:*
% For unstable system: A = [1 2; 0 0.5]
% (a) Try solving Lyapunov equation with Q = I
% (b) Check eigenvalues of resulting P
% (c) Is P positive definite?
% (d) What does this tell you about stability?

% Your code here:


%% 
% *Problem 4.2.3:*
% Discrete-time Lyapunov equation:
% A_d = [0.8 0.1; -0.1 0.9]
% (a) Solve discrete Lyapunov equation using dlyap()
% (b) Check if P is positive definite
% (c) Verify solution: A_d^T*P*A_d - P + Q = 0
% (d) Is discrete system stable?

% Your code here:


%% Exercise 4.3: Matrix Definiteness
% *Problem 4.3.1:*
% Test positive definiteness:
% M = [4 1; 1 3]
% (a) Compute eigenvalues - are they all positive?
% (b) Use Cholesky decomposition: chol(M, 'lower')
% (c) Verify M = L*L^T
% (d) Conclude if M is positive definite

% Your code here:


%% 
% *Problem 4.3.2:*
% For matrix M = [1 2; 2 1]
% (a) Try Cholesky decomposition
% (b) Check eigenvalues
% (c) Is this matrix positive definite?
% (d) Classify the matrix (positive definite/semi-definite/indefinite)

% Your code here:


%% 
% *Problem 4.3.3:*
% Test multiple matrices:
% M1 = [2 -1; -1 2], M2 = [1 2; 2 1], M3 = [3 0; 0 3]
% (a) For each matrix, compute eigenvalues
% (b) Attempt Cholesky decomposition
% (c) Classify each matrix
% (d) Which are suitable as Lyapunov function candidates?

% Your code here:


%% Exercise 4.4: BIBO Stability
% *Problem 4.4.1:*
% Transfer function analysis:
% H(s) = 1/(s² + 3s + 2)
% (a) Create transfer function
% (b) Find poles using pole()
% (c) Check if all poles have negative real parts
% (d) Is system BIBO stable?

% Your code here:


%% 
% *Problem 4.4.2:*
% Compare stable vs unstable:
% H1(s) = (s+1)/(s²+4s+3)
% H2(s) = (s+1)/(s²-4s+3)
% (a) Find poles of both systems
% (b) Determine BIBO stability of each
% (c) Plot step responses
% (d) Observe bounded vs unbounded behavior

% Your code here:


%% 
% *Problem 4.4.3:*
% System with pole-zero cancellation:
% H(s) = (s+2)/(s²+3s+2)
% (a) Factor numerator and denominator
% (b) Identify any cancellations
% (c) Find remaining poles
% (d) Determine BIBO stability

% Your code here:


%% Exercise 4.5: State Feedback - Pole Placement
% *Problem 4.5.1:*
% Design state feedback controller:
% A = [0 1; -2 -3], B = [0; 1]
% Desired poles: [-4, -5]
% (a) Check controllability using ctrb()
% (b) Design gain K using place()
% (c) Compute closed-loop A_cl = A - B*K
% (d) Verify closed-loop eigenvalues match desired poles

% Your code here:


%% 
% *Problem 4.5.2:*
% Stabilize unstable system:
% A = [1 1; 0 2], B = [0; 1]
% (a) Find open-loop eigenvalues - is it stable?
% (b) Check controllability
% (c) Design K to place poles at [-1, -2]
% (d) Plot initial condition response before and after feedback

% Your code here:


%% 
% *Problem 4.5.3:*
% Third-order system:
% A = [0 1 0; 0 0 1; -1 -2 -3], B = [0; 0; 1]
% Desired poles: [-2, -3+2i, -3-2i]
% (a) Verify controllability
% (b) Design state feedback gain K
% (c) Compare open-loop vs closed-loop poles
% (d) Simulate response with x(0) = [1; 0; 0]

% Your code here:


%% Exercise 4.6: Ackermann's Formula
% *Problem 4.6.1:*
% Use Ackermann's formula:
% A = [0 1; -6 -5], B = [0; 1]
% Desired poles: [-2, -3]
% (a) Design K using acker()
% (b) Design K using place()
% (c) Compare both results
% (d) Verify closed-loop poles

% Your code here:


%% 
% *Problem 4.6.2:*
% For system: A = [1 1; 1 2], B = [1; 0]
% (a) Use acker() with poles at [-1, -4]
% (b) Verify controllability first
% (c) Check closed-loop eigenvalues
% (d) Plot step response of closed-loop system

% Your code here:


%% Exercise 4.7: LQR - Linear Quadratic Regulator
% *Problem 4.7.1:*
% Basic LQR design:
% A = [0 1; -1 -0.5], B = [0; 1]
% Q = eye(2), R = 1
% (a) Design LQR controller using lqr()
% (b) Display optimal gain K
% (c) Find closed-loop poles
% (d) Verify they are in left half-plane

% Your code here:


%% 
% *Problem 4.7.2:*
% Effect of Q matrix:
% Same system as 4.7.1
% (a) Design LQR with Q = diag([1, 1])
% (b) Design LQR with Q = diag([10, 1])
% (c) Design LQR with Q = diag([100, 1])
% (d) Compare gains K and closed-loop poles
% (e) Plot responses - which is faster?

% Your code here:


%% 
% *Problem 4.7.3:*
% Effect of R matrix:
% A = [0 1; -2 -1], B = [0; 1], Q = diag([10, 1])
% (a) Design with R = 0.1 (cheap control)
% (b) Design with R = 1
% (c) Design with R = 10 (expensive control)
% (d) Compare control efforts for same initial condition
% (e) Plot state and control trajectories

% Your code here:


%% Exercise 4.8: Riccati Equation
% *Problem 4.8.1:*
% Solve CARE directly:
% A = [0 1; -2 -3], B = [0; 1], Q = eye(2), R = 1
% (a) Solve using care()
% (b) Extract solution P and gain K
% (c) Verify CARE: A^TP + PA - PBR^(-1)B^TP + Q = 0
% (d) Compare with lqr() result

% Your code here:


%% 
% *Problem 4.8.2:*
% Check properties of P:
% For same system as 4.8.1
% (a) Verify P is symmetric
% (b) Check P is positive definite (eigenvalues)
% (c) Use P as Lyapunov function candidate
% (d) Verify stability

% Your code here:


%% Exercise 4.9: Comprehensive Pole Placement
% *Problem 4.9.1:*
% Mass-spring-damper system:
% A = [0 1; -4 -2], B = [0; 1] (force input)
% (a) Find natural frequency and damping from A
% (b) Design feedback for ζ = 0.7, ωn = 3 rad/s
% (c) Compute desired poles from ζ and ωn
% (d) Design K and verify response characteristics

% Your code here:


%% 
% *Problem 4.9.2:*
% DC Motor control:
% A = [-R/L -Kb/L; Kt/J -B/J], B = [1/L; 0]
% Use: R=1, L=0.5, Kt=0.01, Kb=0.01, J=0.01, B=0.1
% (a) Create A and B matrices
% (b) Check open-loop stability
% (c) Design state feedback for poles at [-5, -10]
% (d) Simulate with initial angular velocity

% Your code here:


%% 
% *Problem 4.9.3:*
% Satellite attitude control:
% A = [0 1; 0 0], B = [0; 1] (double integrator)
% (a) Check controllability
% (b) Design for critically damped response: ωn = 1
% (c) Place both poles at -ωn
% (d) Compare with LQR design (Q = diag([10, 1]), R = 1)

% Your code here:


%% Exercise 4.10: LQR Applications
% *Problem 4.10.1:*
% Inverted pendulum on cart (linearized):
% A = [0 1 0 0; 0 0 -m*g/M 0; 0 0 0 1; 0 0 (M+m)*g/(M*L) 0]
% Use: m = 0.1, M = 1, L = 0.5, g = 9.81
% B = [0; 1/M; 0; -1/(M*L)]
% (a) Verify system is unstable (open-loop eigenvalues)
% (b) Design LQR with Q = diag([10, 1, 100, 1]), R = 1
% (c) Check closed-loop stability
% (d) Simulate stabilization from small initial angle

% Your code here:


%% 
% *Problem 4.10.2:*
% Compare pole placement vs LQR:
% A = [0 1; -1 -1], B = [0; 1]
% (a) Pole placement: place at [-2, -3]
% (b) LQR: use Q = diag([5, 1]), R = 1
% (c) Compare resulting gains K
% (d) Compare closed-loop poles
% (e) Which has better performance/control trade-off?

% Your code here:


%% Exercise 4.11: Comprehensive Problems
% *Problem 4.11.1:*
% Complete stability analysis:
% A = [-1 2; -2 -3]
% (a) Eigenvalue-based stability test
% (b) Lyapunov-based stability test
% (c) Compute phase portrait
% (d) All methods should agree - verify

% Your code here:


%% 
% *Problem 4.11.2:*
% Design trade-off study:
% A = [0 1; -2 -1], B = [0; 1]
% (a) Design with LQR for 5 different Q matrices
% (b) For each, compute: settling time, overshoot, control effort
% (c) Plot trade-off: performance vs control cost
% (d) Select best compromise

% Your code here:


%% 
% *Problem 4.11.3:*
% Stabilization challenge:
% A = [2 1 0; 0 1 1; 0 0 -1], B = [1; 0; 1]
% (a) Check open-loop stability
% (b) Verify controllability
% (c) Design stabilizing controller (your choice: pole placement or LQR)
% (d) Prove closed-loop stability
% (e) Simulate with x(0) = [1; 1; 1]

% Your code here:


%% 
% *Problem 4.11.4:*
% Robustness analysis:
% A = [0 1; -2 -3], B = [0; 1]
% Design K with desired poles at [-4, -5]
% (a) Compute nominal closed-loop system
% (b) Add ±20% uncertainty to A: A_new = A*(1+δ)
% (c) For δ = -0.2, 0, 0.2: compute closed-loop poles
% (d) Does system remain stable under uncertainty?

% Your code here:



