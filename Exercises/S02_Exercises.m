%% Season 2 Exercises: State-Space Modeling and Linearization
close all; clear; clc;

% Put the last 2 digits of your student number in the RNG function:
rng(0);

%%% Name :
%%% Student number =

%% Exercise 2.1: From Differential Equations to State-Space
% *Problem 2.1.1:*
% Convert the following second-order ODE to state-space form:
% ẍ + 4ẋ + 3x = 2u
% Choose state variables: x₁ = x, x₂ = ẋ

% (a) Write the state-space matrices A, B, C, D
% (b) Create the state-space system using ss()
% (c) Display the system

% Your code here:


%% 
% *Problem 2.1.2:*
% RLC Circuit: L*d²i/dt² + R*di/dt + (1/C)*i = dv/dt
% Given: L = 1 H, R = 2 Ω, C = 0.5 F
% (a) Define state variables: x₁ = i, x₂ = di/dt
% (b) Derive state-space matrices A and B
% (c) Assume output is current i, find C and D
% (d) Create state-space model

% Your code here:


%% 
% *Problem 2.1.3:*
% Mass-Spring-Damper with two masses:
% m₁ẍ₁ + c₁ẋ₁ + k₁x₁ + k₂(x₁-x₂) = F
% m₂ẍ₂ + c₂ẋ₂ + k₂(x₂-x₁) = 0
% Use m₁=1, m₂=2, c₁=0.5, c₂=0.3, k₁=2, k₂=1
% (a) Define 4 states: [x₁, ẋ₁, x₂, ẋ₂]
% (b) Create 4×4 state matrix A
% (c) Create input matrix B (force applied to mass 1)
% (d) Output is position of mass 2, find C and D

% Your code here:


%% Exercise 2.2: State-Space and Transfer Function Conversions
% *Problem 2.2.1:*
% Given transfer function: H(s) = (s + 2)/(s² + 3s + 2)
% (a) Create transfer function using tf()
% (b) Convert to state-space using tf2ss()
% (c) Display the A, B, C, D matrices
% (d) Convert back to transfer function and verify

% Your code here:


%% 
% *Problem 2.2.2:*
% Given state-space system:
% A = [0 1; -5 -6], B = [0; 1], C = [1 1], D = 0
% (a) Create state-space object
% (b) Convert to transfer function using ss2tf()
% (c) Extract numerator and denominator
% (d) Verify the transfer function is correct

% Your code here:


%% 
% *Problem 2.2.3:*
% Create a third-order system:
% H(s) = (2s + 5)/(s³ + 6s² + 11s + 6)
% (a) Create transfer function
% (b) Find poles and zeros using pole() and zero()
% (c) Convert to state-space
% (d) Verify eigenvalues of A equal the poles

% Your code here:


%% Exercise 2.3: System Response Analysis
% *Problem 2.3.1:*
% For the system: A = [0 1; -4 -5], B = [0; 1], C = [1 0], D = 0
% (a) Compute and plot step response for 10 seconds
% (b) Compute and plot impulse response for 10 seconds
% (c) Use stepinfo() to find rise time, settling time, overshoot
% (d) Display all performance metrics

% Your code here:


%% 
% *Problem 2.3.2:*
% For the same system as above:
% (a) Simulate initial condition response with x(0) = [2; -1]
% (b) Plot the response for 8 seconds
% (c) Extract state trajectory x(t) from the response
% (d) Plot both states x₁(t) and x₂(t) on separate subplots

% Your code here:


%% 
% *Problem 2.3.3:*
% Custom input response:
% Use system: A = [-1 0; 0 -2], B = [1; 1], C = [1 1], D = 0
% (a) Create time vector t = 0:0.01:10
% (b) Create input u(t) = sin(2*t) + 0.5*cos(3*t)
% (c) Use lsim() to simulate the response
% (d) Plot input and output on the same figure

% Your code here:


%% Exercise 2.4: State Transition and Matrix Exponential
% *Problem 2.4.1:*
% For matrix A = [0 1; -3 -4]:
% (a) Compute state transition matrix Φ(t) = exp(A*t) at t = 1, 2, 5
% (b) With x(0) = [1; 0], compute x(t) at these times
% (c) Verify Φ(0) = I (identity matrix)
% (d) Display all results

% Your code here:


%% 
% *Problem 2.4.2:*
% Compare matrix exponential with ODE solver:
% A = [-1 2; -2 -3], x(0) = [2; 1]
% (a) Compute trajectory using expm() for t = 0:0.1:5
% (b) Compute trajectory using ode45() for same time range
% (c) Plot both solutions on same graph
% (d) Calculate maximum difference between solutions

% Your code here:


%% Exercise 2.5: Laplace Transforms
% *Problem 2.5.1:*
% Compute Laplace transforms of:
% (a) f₁(t) = t*exp(-2*t)
% (b) f₂(t) = sin(3*t)*exp(-t)
% (c) f₃(t) = t²*cos(t)
% Verify by computing inverse Laplace transform

% Your code here:


%% 
% *Problem 2.5.2:*
% Given F(s) = (3s + 5)/(s² + 4s + 3)
% (a) Use partial fraction expansion with residue()
% (b) Find residues, poles, and direct term
% (c) Compute inverse Laplace transform using ilaplace()
% (d) Verify the result

% Your code here:


%% Exercise 2.6: Equilibrium Points and Stability
% *Problem 2.6.1:*
% Van der Pol oscillator: ẍ - μ(1-x²)ẋ + x = 0
% With μ = 1, states: x₁ = x, x₂ = ẋ
% (a) Write the nonlinear state equations
% (b) Find equilibrium points (set ẋ = 0)
% (c) How many equilibrium points exist?
% (d) Which are stable?

% Your code here:


%% 
% *Problem 2.6.2:*
% Predator-Prey model:
% ẋ = ax - bxy, 
% ẏ = -cy + dxy
% Use a=1, b=0.1, c=1.5, d=0.075

% (a) Find all equilibrium points
% (b) Verify equilibria by substitution
% (c) Interpret physical meaning of each equilibrium

% Your code here:


%% Exercise 2.7: Jacobian Linearization
% *Problem 2.7.1:*
% For pendulum: θ̈ + (g/L)sin(θ) + b*θ̇ = u
% Use g=9.81, L=1, b=0.5
% States: x = [θ; θ̇], input: u = torque

% (a) Write symbolic state equations
% (b) Compute Jacobian matrices A = ∂f/∂x and B = ∂f/∂u
% (c) Linearize at θ = 0 (hanging down)
% (d) Find eigenvalues - is it stable?

% Your code here:


%% 
% *Problem 2.7.2:*
% Continue with pendulum from 2.7.1:
% (a) Linearize at θ = π (inverted position)
% (b) Compute A and B at this equilibrium
% (c) Find eigenvalues - is it stable?
% (d) Compare eigenvalues with hanging down position

% Your code here:


%% 
% *Problem 2.7.3:*
% Nonlinear system:
% ẋ₁ = x₂
% ẋ₂ = -x₁ + x₁³ - 0.1*x₂ + u
% (a) Define system symbolically
% (b) Compute Jacobian at equilibrium (0, 0)
% (c) Linearize the system
% (d) Analyze stability of linearized system

% Your code here:


%% Exercise 2.8: Multi-State Nonlinear Systems
% *Problem 2.8.1:*
% Three-state system:
% ẋ₁ = x₂
% ẋ₂ = -x₁ - 0.5*x₂ + 2*sin(x₃)
% ẋ₃ = u - 0.2*x₃ + cos(x₁)

% (a) Create symbolic state vector and equations
% (b) Compute 3×3 Jacobian matrix ∂f/∂x
% (c) Compute 3×1 Jacobian matrix ∂f/∂u
% (d) Evaluate at operating point (0, 0, 0, 0)

% Your code here:


%% 
% *Problem 2.8.2:*
% For the system in 2.8.1:
% (a) Create state-space model using linearized A, B
% (b) Find eigenvalues of A
% (c) Is the linearized system stable?
% (d) Create step response plot

% Your code here:


%% Exercise 2.10: Physical Systems Modeling
% DC Motor model:
% Electrical equation: L*di/dt = -R*i - Kb*ω + V
% Mechanical equation: J*dω/dt = Kt*i - B*ω - τ_load
% Parameters: R=1Ω, L=0.5H, Kt=0.01 N·m/A, Kb=0.01 V·s/rad, J=0.01 kg·m², B=0.1 N·m·s

% (a) Define 2 states: [i, ω] (current and angular velocity)
% (b) Create 2×2 state matrix A (assume τ_load = 0)
% (c) Create B matrix (input is voltage V)
% (d) Output is angular velocity ω, create C and D matrices
% (e) Find eigenvalues and check stability
% (f) Plot step response to 1V input

% Your code here:



