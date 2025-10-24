%% Season 3: Controllability, Observability, and Realization
% *Learning Outcomes:*
%% 
% * Test controllability and observability using rank and PBH criteria
% * Understand and compute controllability/observability gramians
% * Perform Kalman decomposition to identify structural properties
% * Transform systems to canonical forms
% * Compute minimal realizations
% * Apply balanced realization for model reduction
%% 
% *Prerequisites:* Season 1 & 2 (Mathematical Foundations, State-Space Modeling)
% 
% *MATLAB Version:* R2025b
% 
% *Toolboxes Required:* Control System Toolbox, Symbolic Math Toolbox

close all; clear; clc;
rng(0);
%% Section 3.1: Controllability - Rank Test
% *Mathematical Background*
% 
% A system (A,B) is controllable if the *controllability matrix* has full rank:
% 
% $$\mathcal{C} = [B\ AB\ A^2B\ \ldots\ A^{n-1}B]$$
% 
% rank(C) = n ⟺ system is controllable
% 
% *Physical Meaning:* Can reach any state from any initial state using input 
% u(t)
% 
% Example 1: Fully controllable system

A1 = [0 1; -2 -3];
B1 = [0; 1];

fprintf('Example 1: Second-order system\n');
fprintf('A =\n'); disp(A1);
fprintf('B =\n'); disp(B1);

% Compute controllability matrix
C_ctrb1 = ctrb(A1, B1);
rank_c1 = rank(C_ctrb1);
n1 = size(A1, 1);

fprintf('Controllability matrix C = [B AB]:\n');
disp(C_ctrb1);
fprintf('Rank: %d, System order: %d\n', rank_c1, n1);

if rank_c1 == n1
    fprintf('✓ System is CONTROLLABLE\n\n');
else
    fprintf('✗ System is NOT controllable\n\n');
end

%% 
% Example 2: Uncontrollable system

A2 = [1 1; 4 -2];
B2 = [0; 0];

fprintf('Example 2: Uncontrollable system\n');
fprintf('A =\n'); disp(A2);
fprintf('B =\n'); disp(B2);

C_ctrb2 = ctrb(A2, B2);
rank_c2 = rank(C_ctrb2);
n2 = size(A2, 1);

fprintf('Controllability matrix C = [B AB]:\n');
disp(C_ctrb2);
fprintf('Rank: %d, System order: %d\n', rank_c2, n2);

if rank_c2 == n2
    fprintf('✓ System is CONTROLLABLE\n\n');
else
    fprintf('✗ System is NOT controllable (rank deficiency: %d)\n\n', n2 - rank_c2);
end

%% 
% Example 3: Multi-input system

A3 = [0 1 0; 0 0 1; -1 -2 -3];
B3 = [0 1; 0 0; 1 0];

fprintf('Example 3: Multi-input system (3 states, 2 inputs)\n');
fprintf('A =\n'); disp(A3);
fprintf('B =\n'); disp(B3);

C_ctrb3 = ctrb(A3, B3);
rank_c3 = rank(C_ctrb3);
n3 = size(A3, 1);

fprintf('Controllability matrix C = [B AB]:\n');disp(C_ctrb3);
fprintf('Controllability matrix size: %dx%d\n', size(C_ctrb3, 1), size(C_ctrb3, 2));
fprintf('Rank: %d, System order: %d\n', rank_c3, n3);

if rank_c3 == n3
    fprintf('✓ System is CONTROLLABLE\n\n');
else
    fprintf('✗ System is NOT controllable\n\n');
end
%% Section 3.2: Observability - Rank Test
% *Mathematical Background*
% 
% A system (A,C) is observable if the observability matrix has full rank:
% 
% $$O=\left\lbrack \begin{array}{c}C\\\textrm{CA}\\\vdots \\{\textrm{CA}}^{N-1} 
% \end{array}\right\rbrack$$
% 
% rank(O) = n ⟺ system is observable
% 
% *Physical Meaning:* Can determine initial state from output measurements
% 
% Example 1: Fully observable system

C1 = [1 0];  % Measure first state

fprintf('Example 1: Measure first state only\n');
fprintf('A =\n'); disp(A1);
fprintf('C =\n'); disp(C1);

O_obsv1 = obsv(A1, C1);
rank_o1 = rank(O_obsv1);

fprintf('Observability matrix O = [C; CA]:\n');
disp(O_obsv1);
fprintf('Rank: %d, System order: %d\n', rank_o1, n1);

if rank_o1 == n1
    fprintf('✓ System is OBSERVABLE\n\n');
else
    fprintf('✗ System is NOT observable\n\n');
end

%% 
% Example 2: Unobservable system

A4 = [1 1; 4 -2];
C4 = [-1 1; 1 -1];  % Both rows are linearly dependent

fprintf('Example 2: Linearly dependent output measurements\n');
fprintf('A =\n'); disp(A4);
fprintf('C =\n'); disp(C4);

O_obsv2 = obsv(A4, C4);
rank_o2 = rank(O_obsv2);
n4 = size(A4, 1);

fprintf('Observability matrix O = [C; CA]:\n');
disp(O_obsv2);
fprintf('Rank: %d, System order: %d\n', rank_o2, n4);

if rank_o2 == n4
    fprintf('✓ System is OBSERVABLE\n\n');
else
    fprintf('✗ System is NOT observable (rank deficiency: %d)\n\n', n4 - rank_o2);
end
%% Section 3.3: PBH (Popov-Belevitch-Hautus) Test
% *Mathematical Background*
% 
% *Controllability PBH Test:*
% 
% (A,B) is controllable ⟺ rank([λI-A B]) = n for all eigenvalues λ
% 
% *Observability PBH Test:*
% 
% (A,C) is observable ⟺ rank([λI-A; C]) = n for all eigenvalues λ

A_pbh = [1 1; 4 -2];
B_pbh = [1; -2];
C_pbh = [1 0];

fprintf('System matrices:\n');
fprintf('A =\n'); disp(A_pbh);
fprintf('B =\n'); disp(B_pbh);
fprintf('C =\n'); disp(C_pbh);

% Get eigenvalues
eigenvalues = eig(A_pbh);
fprintf('Eigenvalues: ');
fprintf('%.4f ', eigenvalues);
fprintf('\n\n');

% PBH controllability test

fprintf('PBH Controllability Test:\n');
fprintf('Checking rank([λI-A, B]) for each eigenvalue λ\n\n');

n_pbh = size(A_pbh, 1);
controllable_pbh = false;

for i = 1:length(eigenvalues)
    lambda = eigenvalues(i);
    M = [lambda*eye(n_pbh) - A_pbh, B_pbh];
    r = rank(M);
    fprintf('λ = %.4f: rank([λI-A, B]) = %d ', lambda, r);
    if r < n_pbh
        fprintf('✗ FAILS\n');
        controllable_pbh = false;
    else
        fprintf('✓\n');
        controllable_pbh = true;
    end
end

if controllable_pbh
    fprintf('\nConclusion: System is CONTROLLABLE\n\n');
else
    fprintf('\nConclusion: System is NOT CONTROLLABLE\n\n');
end

% PBH observability test

fprintf('PBH Observability Test:\n');
fprintf('Checking rank([λI-A; C]) for each eigenvalue λ\n\n');

observable_pbh = false;

for i = 1:length(eigenvalues)
    lambda = eigenvalues(i);
    M = [lambda*eye(n_pbh) - A_pbh; C_pbh];
    r = rank(M);
    fprintf('λ = %.4f: rank([λI-A; C]) = %d ', lambda, r);
    if r < n_pbh
        fprintf('✗ FAILS\n');
        observable_pbh = false;
    else
        fprintf('✓\n');
        observable_pbh = true;
    end
end

if observable_pbh
    fprintf('\nConclusion: System is OBSERVABLE\n\n');
else
    fprintf('\nConclusion: System is NOT OBSERVABLE\n\n');
end
%% Section 3.4: Controllability and Observability Gramians
% *Mathematical Background*
% 
% *Controllability Gramian:* $W_c = \int_0^{\infty} e^{At}BB^Te^{A^Tt}dt$
% 
% Solves: $AW_c + W_cA^T + BB^T = 0$ (Lyapunov equation)
% 
% *Observability Gramian:* $W_o = \int_0^{\infty} e^{A^Tt}C^TCe^{At}dt$
% 
% Solves: $A^TW_o + W_oA + C^TC = 0$
% 
% *Properties:*
%% 
% * System is controllable ⟺ $W_c > 0$ (positive definite)
% * System is observable ⟺ $W_o > 0$ (positive definite)

A_gram = [0 1; -2 -3];
B_gram = [0; 1];
C_gram = [1 0];

fprintf('System:\n');
fprintf('A =\n'); disp(A_gram);
fprintf('B =\n'); disp(B_gram);
fprintf('C =\n'); disp(C_gram);

% Controllability gramian

Wc = gram(ss(A_gram, B_gram, C_gram, 0), 'c');
fprintf('Controllability Gramian Wc:\n');
disp(Wc);

% Check positive definiteness
eig_Wc = eig(Wc);
fprintf('Eigenvalues of Wc: ');
fprintf('%.4f ', eig_Wc);
fprintf('\n');

if all(eig_Wc > 0)
    fprintf('Wc is positive definite → System is CONTROLLABLE\n\n');
else
    fprintf('Wc is NOT positive definite → System is NOT controllable\n\n');
end
% Observability gramian

Wo = gram(ss(A_gram, B_gram, C_gram, 0), 'o');
fprintf('Observability Gramian Wo:\n');
disp(Wo);

eig_Wo = eig(Wo);
fprintf('Eigenvalues of Wo: ');
fprintf('%.4f ', eig_Wo);
fprintf('\n');

if all(eig_Wo > 0)
    fprintf('Wo is positive definite → System is OBSERVABLE\n\n');
else
    fprintf('Wo is NOT positive definite → System is NOT observable\n\n');
end
%% Section 3.5: Kalman Decomposition
% *Concept Overview*
% 
% Kalman decomposition separates a system into four subsystems:
%% 
% # Controllable and observable
% # Controllable but not observable
% # Not controllable but observable
% # Not controllable and not observable

A_kd = [-1  0   0   0;
         0  -2  0   0;
         0   0  -3  0;
         0   0   0  -4];
     
B_kd = [1; 1; 0; 0];  % States 1,2 controllable; 3,4 not controllable
C_kd = [1  0  1  0];  % States 1,3 observable; 2,4 not observable

fprintf('Original system (4 states):\n');
fprintf('A =\n'); disp(A_kd);
fprintf('B =\n'); disp(B_kd);
fprintf('C =\n'); disp(C_kd);

% Check controllability and observability
Cc = ctrb(A_kd, B_kd);
Oc = obsv(A_kd, C_kd);

fprintf('Controllability rank: %d / %d\n', rank(Cc), size(A_kd,1));
fprintf('Observability rank: %d / %d\n\n', rank(Oc), size(A_kd,1));

% Create state-space model
sys_full = ss(A_kd, B_kd, C_kd, 0);
sys_min = minreal(sys_full);

fprintf('Minimal realization (removes uncontrollable/unobservable modes):\n');
fprintf('Original system: %d states\n', size(sys_full.A, 1));
fprintf('Minimal realization: %d states\n', size(sys_min.A, 1));
fprintf('Reduction: %d states removed\n\n', size(sys_full.A,1) - size(sys_min.A,1));

% Compare transfer functions
fprintf('Transfer function (original):\n');
[num_full, den_full] = ss2tf(A_kd, B_kd, C_kd, 0)
fprintf('Transfer function (minimal):\n');
[num_min, den_min] = ss2tf(sys_min.A, sys_min.B, sys_min.C, sys_min.D)

% fprintf('Transfer function (original):\n');
% disp(tf(num_full, den_full));
% 
% fprintf('Transfer function (minimal):\n');
% disp(tf(num_min, den_min));

% Verify step responses are identical
figure('Name', 'Kalman Decomposition - Step Response Comparison');
t = 0:0.01:5;
[y_full, ~] = step(sys_full, t);
[y_min, ~] = step(sys_min, t);

plot(t, y_full, 'b-', 'LineWidth', 2, 'DisplayName', 'Full (4 states)');
hold on;
plot(t, y_min, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Minimal (1 state)');
grid on;
xlabel('Time (s)');
ylabel('Output');
title('Step Response: Full vs Minimal Realization');
legend('Location', 'best');

fprintf('Step response difference (max error): %.2e\n\n', max(abs(y_full - y_min)));
%% Section 3.6: Canonical Forms
% *Concept Overview*
% 
% Canonical forms are standard state-space representations:
%% 
% * *Controllable canonical form* - easy controller design
% * *Observable canonical form* - easy observer design
% * *Modal (diagonal) canonical form* - decoupled dynamics
%% 
% Function: |canon(sys, 'type')|

% Original system
A_orig = [0 1; -6 -5];
B_orig = [0; 1];
C_orig = [1 0];
D_orig = 0;

sys_orig = ss(A_orig, B_orig, C_orig, D_orig);

fprintf('Original system:\n');
fprintf('A =\n'); disp(A_orig);
fprintf('B =\n'); disp(B_orig);
%% 
% Controllable canonical form (companion form)

[sys_ctrl, T_ctrl] = canon(sys_orig,'companion');

fprintf('Controllable Canonical Form:\n');
fprintf('A_c =\n'); disp(sys_ctrl.A);
fprintf('B_c =\n'); disp(sys_ctrl.B);
fprintf('Transformation matrix T:\n'); disp(T_ctrl);
%% 
% *Observable canonical form*
% 
% Observable companion via dual + map back

% 1) Dual system
sys_dual = ss(A_orig.', C_orig.', B_orig.', D_orig);

% 2) Put the dual in controllable companion
[sys_dual_c, T_dual] = canon(sys_dual, 'companion');

% 3) Map back to observable companion of the original
A_o = sys_dual_c.A.';     % transpose back
B_o = sys_dual_c.C.';     % note the swap
C_o = sys_dual_c.B.';
D_o = D_orig;

% 4) Transformation for the original system (so that T_o^{-1} A T_o = A_o)
T_o = inv(T_dual.');      % T_o = (T_dual^{-1})^T

%% Display
fprintf('Observable Canonical Form:\n');
disp('A_o ='); disp(A_o);
disp('B_o ='); disp(B_o);
disp('C_o ='); disp(C_o);
disp('Transformation matrix T_o ='); disp(T_o);

%% Verify
tol = 1e-10;
ok = norm(T_o\A_orig*T_o - A_o, 'fro') < tol && ...
     norm(T_o\B_orig - B_o)     < tol   && ...
     norm(C_orig*T_o - C_o)     < tol;
fprintf('Verification passed: %d\n', ok);

%% 
% Modal (diagonal) form

[sys_modal, T_modal] = canon(sys_orig, 'modal');

fprintf('\nModal Canonical Form:\n');
fprintf('A_m (diagonal with eigenvalues):\n'); 
disp(sys_modal.A);
fprintf('B_m =\n'); disp(sys_modal.B);
%% 
% Verify eigenvalues are preserved

fprintf('Eigenvalues (original): ');
fprintf('%.4f ', eig(A_orig));
fprintf('\n');
fprintf('Eigenvalues (modal): ');
fprintf('%.4f ', eig(sys_modal.A));
fprintf('\n\n');
%% Section 3.7: State-Space Transformations
% *Mathematical Background*
% 
% Similarity transformation: $\bar{A} = T^{-1}AT$, $\bar{B} = T^{-1}B$, $\bar{C} 
% = CT$
% 
% Transfer function is invariant: $C(sI-A)^{-1}B = \bar{C}(sI-\bar{A})^{-1}\bar{B}$

A_tf = [0 1; -2 -3];
B_tf = [0; 1];
C_tf = [1 0];
D_tf = 0;

% Custom transformation matrix
T = [1 2; 0 1];

fprintf('Original system:\n');
fprintf('A =\n'); disp(A_tf);
fprintf('B =\n'); disp(B_tf);
fprintf('C =\n'); disp(C_tf);

fprintf('\nTransformation matrix T:\n');
disp(T);

% Transform using ss2ss
sys_original = ss(A_tf, B_tf, C_tf, D_tf);
sys_transformed = ss2ss(sys_original, T); % ss2ss performs the similarity transformation

fprintf('Transformed system (using ss2ss):\n');
fprintf('A_new =\n'); disp(sys_transformed.A);
fprintf('B_new =\n'); disp(sys_transformed.B);
fprintf('C_new =\n'); disp(sys_transformed.C);

% Manual transformation for verification
A_new_manual = T * A_tf / T;
B_new_manual = T * B_tf;
C_new_manual = C_tf / T;

fprintf('\nManual transformation verification:\n');
fprintf('A_new (manual) =\n'); disp(A_new_manual);
fprintf('Error in A: %.2e\n', norm(sys_transformed.A - A_new_manual));

% Verify transfer functions are identical
fprintf('\nTransfer function (original):\n');
tf_orig = tf(sys_original)
fprintf('Transfer function (transformed):\n');
tf_trans = tf(sys_transformed)
%% Section 3.8: Balanced Realization
% *Concept Overview*
% 
% Balanced realization transforms system so that:
%% 
% * Controllability gramian = Observability gramian = Σ (diagonal)
% * Diagonal elements (Hankel singular values) indicate state importance
% * Enables systematic model reduction
%% 
% Function: |balreal(sys)|

% Create a higher-order system
A_bal = [-1  0   0.5  0;
          0  -2  0    0.3;
          0  0   -5   0;
          0  0   0    -10];
      
B_bal = [1; 0.5; 0.2; 0.1];
C_bal = [1 1 0.5 0.1];
D_bal = 0;

sys_unbal = ss(A_bal, B_bal, C_bal, D_bal);

fprintf('Original system (4 states):\n');

% Compute gramians before balancing
Wc_before = gram(sys_unbal, 'c');
Wo_before = gram(sys_unbal, 'o');

fprintf('Controllability gramian eigenvalues:\n');
disp(sort(eig(Wc_before), 'descend'));

fprintf('Observability gramian eigenvalues:\n');
disp(sort(eig(Wo_before), 'descend'));

% Balanced realization
[sys_bal, g, T_bal] = balreal(sys_unbal);

fprintf('\nHankel singular values (importance of each state):\n');
disp(g);

fprintf('Balanced system:\n');
sys_bal

% Gramians of balanced system should be equal and diagonal
Wc_after = gram(sys_bal, 'c');
Wo_after = gram(sys_bal, 'o');

fprintf('Controllability gramian (balanced):\n');
disp(Wc_after);

fprintf('Observability gramian (balanced):\n');
disp(Wo_after);

fprintf('Difference between gramians: %.2e\n', norm(Wc_after - Wo_after));

%% 
% Model reduction: keep only significant states
% 
% If a Hankel singular value is very small, that state contributes little

threshold = 0.1 * g(1);  % Keep states with σ > 10% of largest
n_keep = sum(g > threshold);

fprintf('\nModel reduction threshold: %.4f\n', threshold);
fprintf('States to keep: %d out of %d\n', n_keep, length(g));

% Extract reduced model
A_red = sys_bal.A(1:n_keep, 1:n_keep);
B_red = sys_bal.B(1:n_keep, :);
C_red = sys_bal.C(:, 1:n_keep);
D_red = sys_bal.D;

sys_reduced = ss(A_red, B_red, C_red, D_red);

fprintf('\nReduced system:\n');
disp(sys_reduced);

% Compare step responses
figure('Name', 'Balanced Realization - Model Reduction');
step(sys_unbal, 'b-', sys_reduced, 'r--', 5);
grid on;
legend('Original (4 states)', sprintf('Reduced (%d states)', n_keep), ...
    'Location', 'best');
title('Step Response: Original vs Reduced Model');
%% 
% using a task:

%% Reduce LTI model order using balanced truncation
 
% Compute reduced-order approximation
R = reducespec(sys_unbal,'balanced');
% Compute MOR data once
R = process(R);
% Get reduced-order model
sysReduced = getrom(R,Order=1,Method='truncate');
 
% Create comparison plot
f = figure();
h = stepplot(f,sys_unbal,sysReduced);
legend(h,['Original Model (',mat2str(order(sys_unbal)),' states)'],...
         ['Reduced Model (',mat2str(order(sysReduced)),' states)']);
h.AxesStyle.GridVisible = true;
 
 
% Remove temporary variables from Workspace
clear R f h
%% Section 3.9: Minimal Realization from Transfer Function
% *Concept Overview*
% 
% Given a transfer function, find the minimal state-space realization:
%% 
% # Convert to state-space: |tf2ss|
% # Remove uncontrollable/unobservable modes: |minreal in which| eliminates 
% uncontrollable or unobservable state in state-space models, or cancels pole-zero 
% pairs in transfer functions or zero-pole-gain models.

% Transfer function with pole-zero cancellation
%% 
% Goal transfer function: $\frac{\left(s+1\right)\left(s+2\right)}{\left(s+1\right)\left(s+3\right)\left(s+4\right)}\longrightarrow 
% \frac{\left(s+2\right)}{\left(s+3\right)\left(s+4\right)}$

num_cancel = conv([1 1], [1 2]);  % (s+1)(s+2)
den_cancel = conv([1 1], conv([1 3], [1 4]));  % (s+1)(s+3)(s+4)

fprintf('Transfer function with cancellation:\n');
fprintf('Numerator (zeros at s=-1,-2): ');
fprintf('%.0f ', num_cancel);
fprintf('\n');
fprintf('Denominator (poles at s=-1,-3,-4): ');
fprintf('%.0f ', den_cancel);
fprintf('\n\n');

fprintf('G(s) =\n');
tf_cancel = tf(num_cancel, den_cancel)

% Convert to state-space (may include cancelling mode)
sys_ss_cancel = ss(tf_cancel);

fprintf('State-space realization (before minreal): %d states\n', size(sys_ss_cancel.A, 1));

%% 
% Minimal realization

sys_min_cancel = minreal(sys_ss_cancel);

fprintf('Minimal realization: %d states\n', size(sys_min_cancel.A, 1));

fprintf('\nMinimal state-space:\n');
fprintf('A =\n'); disp(sys_min_cancel.A);
fprintf('B =\n'); disp(sys_min_cancel.B);

fprintf('Poles (eigenvalues of A): ');
fprintf('%.4f ', eig(sys_min_cancel.A));
fprintf('\n');
fprintf('(Cancelled pole at s=-1 removed)\n\n');
fprintf('Simplified transfer function after minreal:\n');
tf_min_cancel = tf(sys_min_cancel)
%% Section 3.10: Summary and Key Takeaways
% *Key Concepts Covered:*
%% 
% # Controllability: rank test, PBH test, gramians
% # Observability: rank test, PBH test, gramians
% # Kalman decomposition and minimal realization
% # Canonical forms: controllable, observable, modal
% # State-space transformations and similarity
% # Balanced realization for model reduction
% # Minimal realization from transfer functions
%% 
% *MATLAB Functions Mastered:*
% 
% |ctrb|, |obsv|, |rank|, |gram|, |canon|, |ss2ss|, |minreal|, |balreal|, |ss2tf|, 
% |tf2ss|, |ssdata|
% 
% *Structural Property Tests:*
%% 
% * Rank test: $rank(\mathcal{C}) = n$ or $rank(\mathcal{O}) = n$
% * PBH test: $rank([\lambda I-A, B]) = n$ for all eigenvalues
% * Gramian test: $W_c > 0$ or $W_o > 0$
%% 
% *Next Steps:*
% 
% These structural properties enable:
%% 
% * Stability analysis and state feedback design (Season 4)
% * Observer design (requires observability) (Season 5)
% * System simplification and model reduction