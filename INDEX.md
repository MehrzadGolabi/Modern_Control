# MATLAB Modern Control Series - Concept Index

This document provides a comprehensive mapping of modern control theory concepts to specific scripts and sections where they are taught.

## Table of Contents

- [MATLAB Functions Quick Reference](#matlab-functions-quick-reference)
- [Concepts by Category](#concepts-by-category)
- [Detailed Concept Mapping](#detailed-concept-mapping)
- [Quick Lookup by Application](#quick-lookup-by-application)
---

## MATLAB Functions Quick Reference

| Function | Purpose | Primary Location | Alternative Locations |
|:---------|:--------|:-----------------|:---------------------|
| `eig` | Eigenvalues and eigenvectors | S01 §1.6 | S04 §4.1 |
| `inv` | Matrix inverse | S01 §1.2 | - |
| `det` | Determinant | S01 §1.2 | - |
| `rank` | Matrix rank | S01 §1.2 | S03 §3.1, §3.2 |
| `trace` | Matrix trace | S01 §1.2 | - |
| `norm` | Vector/matrix norms | S01 §1.3 | - |
| `null` | Null space | S01 §1.5 | - |
| `orth` | Orthonormal basis | S01 §1.5 | - |
| `dot` | Dot product | S01 §1.5 | - |
| `linsolve` | Solve linear equations | S01 §1.5 | - |
| `triu`, `tril` | Triangular matrices | S01 §1.4 | - |
| `poly` | Characteristic polynomial | S01 §1.6 | - |
| `jordan` | Jordan normal form | S01 §1.8 | - |
| `expm` | Matrix exponential | S01 §1.9 | S02 §2.4 |
| `ss` | State-space model | S02 §2.1 | Throughout |
| `tf` | Transfer function | S02 §2.2 | Throughout |
| `ss2tf` | State-space to transfer function | S02 §2.2 | S03 §3.5 |
| `tf2ss` | Transfer function to state-space | S02 §2.2 | S03 §3.9 |
| `c2d` | Continuous to discrete | S02 §2.6 | S04 §4.10 |
| `d2c` | Discrete to continuous | S02 §2.6 | S04 §4.10 |
| `step` | Step response | S02 §2.3 | Throughout |
| `impulse` | Impulse response | S02 §2.3 | - |
| `initial` | Initial condition response | S02 §2.3 | - |
| `lsim` | Arbitrary input response | S02 §2.3 | - |
| `ode45` | Numerical ODE solver | S02 §2.4 | - |
| `jacobian` | Symbolic Jacobian | S02 §2.8 | S02 §2.9 |
| `laplace` | Laplace transform | S02 §2.5 | - |
| `ilaplace` | Inverse Laplace | S02 §2.5 | - |
| `subs` | Symbolic substitution | S02 §2.8 | - |
| `stepinfo` | Step response metrics | S02 §2.3 | - |
| `residue` | Partial fraction expansion | S02 §2.5 | - |
| `ctrb` | Controllability matrix | S03 §3.1 | - |
| `obsv` | Observability matrix | S03 §3.2 | - |
| `gram` | Controllability/observability gramians | S03 §3.4 | - |
| `canon` | Canonical forms | S03 §3.6 | - |
| `ss2ss` | State transformation | S03 §3.7 | - |
| `minreal` | Minimal realization | S03 §3.5 | S03 §3.9 |
| `balreal` | Balanced realization | S03 §3.8 | - |
| `lyap` | Continuous Lyapunov equation | S04 §4.2 | - |
| `dlyap` | Discrete Lyapunov equation | S04 §4.4 | - |
| `chol` | Cholesky decomposition | S04 §4.3 | - |
| `place` | Pole placement | S04 §4.6 | - |
| `acker` | Ackermann's formula | S04 §4.7 | - |
| `lqr` | Linear quadratic regulator | S04 §4.8 | - |
| `dlqr` | Discrete LQR | S04 §4.10 | - |
| `care` | Continuous Riccati equation | S04 §4.9 | - |
| `pole` | System poles | S04 §4.5 | - |

---

## Concepts by Category

### 1. Mathematical Foundations

| Concept | MATLAB Function | Script | Section |
|:--------|:----------------|:-------|:--------|
| Vector operations | Basic ops | S01 | §1.1 |
| Matrix multiplication | `*`, `.*` | S01 | §1.1 |
| Matrix transpose | `.'`, `'` | S01 | §1.1 |
| Determinant | `det` | S01 | §1.2 |
| Matrix rank | `rank` | S01 | §1.2 |
| Matrix trace | `trace` | S01 | §1.2 |
| Matrix inverse | `inv` | S01 | §1.2 |
| Condition number | `cond` | S01 | §1.2 |
| Vector norms | `norm` | S01 | §1.3 |
| Matrix norms (1, 2, ∞, Fro) | `norm` | S01 | §1.3 |
| Triangular matrices | `triu`, `tril` | S01 | §1.4 |
| Null space | `null` | S01 | §1.5 |
| Orthogonal basis | `orth` | S01 | §1.5 |
| Dot product | `dot` | S01 | §1.5 |
| Linear equation solving | `\`, `linsolve` | S01 | §1.5 |
| Eigenvalues | `eig` | S01 | §1.6 |
| Eigenvectors | `eig` | S01 | §1.6 |
| Characteristic polynomial | `poly` | S01 | §1.6 |
| Similarity transformation | Manual | S01 | §1.7 |
| Jordan normal form | `jordan` | S01 | §1.8 |
| Matrix exponential | `expm` | S01 | §1.9 |

### 2. System Modeling

| Concept | MATLAB Function | Script | Section |
|:--------|:----------------|:-------|:--------|
| ODE to state-space | Manual | S02 | §2.1 |
| State-space object | `ss` | S02 | §2.1, §2.2 |
| Transfer function object | `tf` | S02 | §2.2 |
| SS to TF conversion | `ss2tf` | S02 | §2.2 |
| TF to SS conversion | `tf2ss` | S02 | §2.2 |
| Step response | `step` | S02 | §2.3 |
| Impulse response | `impulse` | S02 | §2.3 |
| Initial condition response | `initial` | S02 | §2.3 |
| General response | `lsim` | S02 | §2.3 |
| Step response metrics | `stepinfo` | S02 | §2.3 |
| State transition matrix | `expm` | S02 | §2.4 |
| ODE simulation | `ode45` | S02 | §2.4 |
| Laplace transform | `laplace` | S02 | §2.5 |
| Inverse Laplace | `ilaplace` | S02 | §2.5 |
| Partial fractions | `residue` | S02 | §2.5 |
| Continuous to discrete | `c2d` | S02 | §2.6 |
| Discrete to continuous | `d2c` | S02 | §2.6 |
| Equilibrium points | Symbolic | S02 | §2.7 |
| Jacobian linearization | `jacobian` | S02 | §2.8 |
| Symbolic substitution | `subs` | S02 | §2.8, §2.9 |
| Physical system examples | Manual | S02 | §2.10 |

### 3. Structural Properties

| Concept | MATLAB Function | Script | Section |
|:--------|:----------------|:-------|:--------|
| Controllability matrix | `ctrb` | S03 | §3.1 |
| Controllability rank test | `rank(ctrb(...))` | S03 | §3.1 |
| Observability matrix | `obsv` | S03 | §3.2 |
| Observability rank test | `rank(obsv(...))` | S03 | §3.2 |
| PBH controllability test | Manual | S03 | §3.3 |
| PBH observability test | Manual | S03 | §3.3 |
| Controllability gramian | `gram(..., 'c')` | S03 | §3.4 |
| Observability gramian | `gram(..., 'o')` | S03 | §3.4 |
| Gramian positive definiteness | `eig` | S03 | §3.4 |
| Kalman decomposition | Manual/`minreal` | S03 | §3.5 |
| Minimal realization | `minreal` | S03 | §3.5, §3.9 |
| Controllable canonical form | `canon(...,'companion')` | S03 | §3.6 |
| Observable canonical form | Dual transform | S03 | §3.6 |
| Modal canonical form | `canon(...,'modal')` | S03 | §3.6 |
| State transformation | `ss2ss` | S03 | §3.7 |
| Balanced realization | `balreal` | S03 | §3.8 |
| Hankel singular values | `balreal` | S03 | §3.8 |
| Model reduction | `balreal` + truncation | S03 | §3.8 |

### 4. Stability and Control

| Concept | MATLAB Function | Script | Section |
|:--------|:----------------|:-------|:--------|
| Eigenvalue stability (CT) | `eig` | S04 | §4.1 |
| Eigenvalue stability (DT) | `eig` | S04 | §4.1 |
| Lyapunov stability theory | Theory | S04 | §4.2 |
| Lyapunov equation (CT) | `lyap` | S04 | §4.2 |
| Lyapunov equation (DT) | `dlyap` | S04 | §4.4 |
| Matrix definiteness | `chol`, `eig` | S04 | §4.3 |
| Cholesky decomposition | `chol` | S04 | §4.3 |
| BIBO stability | `pole` | S04 | §4.5 |
| State feedback | Theory | S04 | §4.6 |
| Pole placement | `place` | S04 | §4.6 |
| Ackermann's formula | `acker` | S04 | §4.7 |
| LQR optimal control | `lqr` | S04 | §4.8 |
| Cost function tuning (Q, R) | Experimentation | S04 | §4.8 |
| Continuous Riccati (CARE) | `care` | S04 | §4.9 |
| Discrete Riccati (DARE) | `dare` | S04 | §4.9 |
| Digital controller design | `dlqr` | S04 | §4.10 |
| Controller discretization | `c2d` | S04 | §4.10 |


## Detailed Concept Mapping

### Comprehensive Concept Coverage

Below is the complete mapping of **all required concepts** from the syllabus:

| Concept | MATLAB Function/Tool | Script | Section | Notes |
|:--------|:--------------------|:-------|:--------|:------|
| **Vector/matrix operations** | Basic ops, `*`, `.*`, `.^` | S01 | §1.1 | Element-wise and matrix operations |
| **Eigenvalues** | `eig` | S01 | §1.6 | Eigenvalues and eigenvectors |
| **Similarity transformation** | `inv`, manual | S01 | §1.7 | $B = T^{-1}AT$ |
| **Linear equation solutions** | `\`, `linsolve` | S01 | §1.5 | Ax = b |
| **Basis and orthonormality** | `null`, `orth`, `dot`, `norm` | S01 | §1.5 | Null space, orthogonal basis |
| **State-space modeling (CT)** | `ss` | S02 | §2.1 | Continuous-time systems |
| **State-space modeling (DT)** | `ss`, `c2d` | S02 | §2.6 | Discrete-time systems |
| **Transfer function ↔ SS** | `ss2tf`, `tf2ss` | S02 | §2.2 | Conversion between representations |
| **State trajectory** | `expm`, `ode45`, `initial` | S02 | §2.4 | Time evolution of states |
| **Controllability** | `ctrb`, `rank` | S03 | §3.1 | Rank test and PBH test |
| **Observability** | `obsv`, `rank` | S03 | §3.2 | Rank test and PBH test |
| **Kalman decomposition** | Custom/`minreal` | S03 | §3.5 | Structural decomposition |
| **Minimal realization** | `minreal`, `balreal` | S03 | §3.5, §3.8 | Remove uncontrollable/unobservable modes |
| **Canonical forms** | `canon`, `ss2ss` | S03 | §3.6 | Controllable, observable, modal forms |
| **Stability (matrix)** | `eig` | S04 | §4.1 | Eigenvalue criterion |
| **Lyapunov stability** | `lyap`, `dlyap` | S04 | §4.2, §4.4 | Lyapunov equation |
| **BIBO stability** | `pole` | S04 | §4.5 | Transfer function poles |
| **Definiteness tests** | `chol`, `eig` | S04 | §4.3 | Positive definite matrices |
| **State feedback** | Theory | S04 | §4.6 | u = -Kx |
| **Pole placement** | `place`, `acker` | S04 | §4.6, §4.7 | Eigenvalue assignment |
| **LQR optimal control** | `lqr` | S04 | §4.8 | Minimize J with Q, R |
| **Riccati equation** | `care`, `dare` | S04 | §4.9 | Algebraic Riccati equation |
| **Simulation** | `ode45`, `lsim`, `sim` | S02 | §2.4 | Simulate control systems |
| **Discretization** | `c2d`, `d2c` | S02, S04 | §2.6, §4.10 | ZOH, Tustin, matched |
| **Digital control** | `dlqr`, `c2d` | S04 | §4.10 | Discrete controller design |
| **Model realization** | `tf2ss`, `minreal`, `canon` | S03 | §3.6, §3.9 | Different realizations |
| **Gain tuning** | `lqr`, experiments | S04 | §4.8 | Q, R matrix selection |
| **Physical examples** | Scripts | S02 | §2.10 | Pendulum, motor, etc. |

---


## Quick Lookup by Application

### For Specific Tasks

| Task | Script | Functions | Notes |
|:-----|:-------|:----------|:------|
| Model a system from physics | S02 | `ss`, `jacobian` | Derive state-space from ODEs |
| Check if controllable | S03 | `ctrb`, `rank` | Rank test |
| Check if observable | S03 | `obsv`, `rank` | Rank test |
| Simplify a model | S03 | `minreal`, `balreal` | Remove extra states |
| Check stability | S04 | `eig`, `lyap` | Multiple methods |
| Design controller | S04 | `place`, `lqr` | Pole placement or optimal |
| Implement digitally | S04 | `c2d`, `dlqr` | Discretize for computer |

---

## Cross-Reference by MATLAB Function

| Function | Primary Use | Example Call | Where Taught |
|:---------|:------------|:-------------|:-------------|
| `eig(A)` | Eigenvalues of A | `lambda = eig(A)` | S01 §1.6 |
| `expm(A*t)` | Matrix exponential | `Phi = expm(A*t)` | S01 §1.9 |
| `ss(A,B,C,D)` | Create state-space | `sys = ss(A,B,C,D)` | S02 §2.1 |
| `ctrb(A,B)` | Controllability matrix | `Co = ctrb(A,B); rank(Co)` | S03 §3.1 |
| `obsv(A,C)` | Observability matrix | `Ob = obsv(A,C); rank(Ob)` | S03 §3.2 |
| `place(A,B,p)` | Pole placement | `K = place(A,B,poles)` | S04 §4.6 |
| `lqr(A,B,Q,R)` | LQR controller | `[K,S,e] = lqr(A,B,Q,R)` | S04 §4.8 |
| `lyap(A,Q)` | Solve Lyapunov eqn | `P = lyap(A',Q)` | S04 §4.2 |

---

**Navigation:**
- [Back to README](README.md)
- [Season 1: Mathematical Foundations](/Matlab%20Live%20Script%20Files/S01_Mathematical_Foundations.mlx)
- [Season 2: State-Space Modeling](/Matlab%20Live%20Script%20Files/S02_StateSpace_Modeling_Linearization.mlx)
- [Season 3: Controllability & Observability](/Matlab%20Live%20Script%20Files/S03_Controllability_Observability_Realization.mlx)
- [Season 4: Stability & Control](/Matlab%20Live%20Script%20Files/S04_Stability_Feedback_LQR.mlx)

---

