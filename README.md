# MATLAB Teaching Series for Modern Control Theory

A comprehensive educational series of MATLAB scripts designed to teach modern control theory from mathematical foundations to complete observer-based controller implementation.

##  Series Overview

This teaching series consists of **5 seasons** (modules), each building upon the previous one to develop a complete understanding of modern control theory and its implementation in MATLAB.

| Season | Title | Topics | Scripts |
|:------:|:------|:-------|:--------|
| **S01** | Mathematical Foundations | Matrix operations, eigenanalysis, norms, Jordan form, matrix exponential | `S01_Mathematical_Foundations.mlx` |
| **S02** | State-Space Modeling & Linearization | ODEs to state-space, Jacobian linearization, Laplace transforms, system responses | `S02_StateSpace_Modeling_Linearization.mlx` |
| **S03** | Controllability, Observability & Realization | Rank tests, PBH test, gramians, Kalman decomposition, canonical forms, minimal realization | `S03_Controllability_Observability_Realization.mlx` |
| **S04** | Stability, Feedback & LQR | Lyapunov theory, BIBO stability, pole placement, LQR, Riccati equations, digital control | `S04_Stability_Feedback_LQR.mlx` |


##  Learning Objectives

By completing this series, you will be able to:

- Represent physical systems in state-space form
- Linearize nonlinear systems around operating points
- Analyze system controllability and observability
- Assess stability using multiple criteria
- Design state feedback controllers (pole placement and LQR)
- Design full and reduced-order observers
- Implement complete observer-based control systems


## 🔧 Requirements

### Software
- **MATLAB**: R2025b or later (scripts may work on R2020a+)
- **Required Toolboxes**:
  - Control System Toolbox
  - Symbolic Math Toolbox

### Prerequisites
- **Mathematical Background**:
  - Linear algebra (matrices, eigenvalues, vector spaces)
  - Ordinary differential equations
  - Basic calculus
  
- **MATLAB Skills**:
  - Basic programming (variables, loops, functions)
  - Plotting and visualization
  - Matrix operations

##  How to Use This Series

### Recommended Learning Path

The seasons are designed to be completed **sequentially**:

```
S01 (Foundations) → S02 (Modeling) → S03 (Structure) → S04 (Control)
```

### For Each Season:

1. **Read the header** - Learning outcomes and prerequisites
2. **Run each section** - Execute code sequentially (click "Run Section" in MATLAB)
3. **Study the output** - Observe numerical results and plots
4. **Read the explanations** - Understand theory behind implementation
5. **Try the exercises** - Practice problems at the end
6. **Experiment** - Modify parameters and observe effects





## 🔑 Key MATLAB Functions by Season

### S01 - Mathematical Foundations
`eig`, `inv`, `det`, `rank`, `trace`, `norm`, `null`, `orth`, `poly`, `jordan`, `expm`, `linsolve`

### S02 - Modeling & Linearization
`ss`, `tf`, `ss2tf`, `tf2ss`, `c2d`, `d2c`, `step`, `impulse`, `initial`, `lsim`, `jacobian`, `laplace`, `ilaplace`, `ode45`

### S03 - Controllability & Observability
`ctrb`, `obsv`, `rank`, `gram`, `canon`, `ss2ss`, `minreal`, `balreal`

### S04 - Stability & Feedback
`lyap`, `dlyap`, `chol`, `place`, `acker`, `lqr`, `dlqr`, `care`, `dare`, `pole`


## 📊 Concept Coverage

All modern control theory concepts are covered comprehensively:

| Category | Concepts | MATLAB Tools |
|:---------|:---------|:-------------|
| **Math Foundation** | Eigenvalues, similarity, norms, Jordan form | `eig`, `norm`, `jordan`, `expm` |
| **System Modeling** | State-space, transfer functions, linearization | `ss`, `tf`, `jacobian`, `c2d` |
| **Structural Properties** | Controllability, observability, realization | `ctrb`, `obsv`, `canon`, `minreal` |
| **Stability** | Eigenvalue, Lyapunov, BIBO | `eig`, `lyap`, `pole` |
| **Control Design** | Pole placement, LQR, digital control | `place`, `lqr`, `dlqr`, `care` |

## Related Resources

### Recommended Textbooks
- **Modern Control Fundamentals** by A. Khaki Sedigh
- **Modern Control Engineering** by Ogata
- **MATLAB for control system engineers** by Rao V Dukkipati


### MATLAB Documentation
- [Control System Toolbox Documentation](https://www.mathworks.com/help/control/)



## 🤝 Support and Feedback

### Getting Help
- Review previous seasons if concepts are unclear
- Check MATLAB documentation (`help function_name`)
- Experiment with simpler examples first
- Use MATLAB's debugging tools (`dbstop`, breakpoints)

### Reporting Issues
If you find errors or have suggestions:
- Document the issue (script name, section, error message)
- Provide MATLAB version and toolbox versions
- Suggest improvements or clarifications
- Open a pull request

## Acknowledgments

This series integrates concepts from:
- Modern control theory literature
- MATLAB Control System documentation
- Feedback from fellow electrical engineering students (Thanks to A. Feizbakhsh and S. Aligholizade for the first drafts and feedbacks)
- A. Moradi Amani, Negahi be Karbord-e Narmafzar MATLAB dar Kontrol-e Modarn (نگاهی به کاربرد نرم افزار متلب در کنترل مدرن), appendix to Modarn Control Fundamentals, by A. Khaki Sedigh.
