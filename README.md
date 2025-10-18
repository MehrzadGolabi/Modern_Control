# MATLAB for Modern Control Theory

A comprehensive educational series of MATLAB scripts designed to teach modern control theory from mathematical foundations to complete observer-based controller implementation.

##  Series Overview

This teaching series consists of **4 seasons** (modules), each building upon the previous one to develop a complete understanding of modern control theory and its implementation in MATLAB.

| Season | Title | Topics | Scripts |
|:------:|:------|:-------|:--------|
| **S01** | Mathematical Foundations | Matrix operations, eigenanalysis, norms, Jordan form, matrix exponential | `S01_Mathematical_Foundations.mlx` |
| **S02** | State-Space Modeling & Linearization | ODEs to state-space, Jacobian linearization, Laplace transforms, system responses | `S02_StateSpace_Modeling_Linearization.mlx` |
| **S03** | Controllability, Observability & Realization | Rank tests, PBH test, gramians, Kalman decomposition, canonical forms, minimal realization | `S03_Controllability_Observability_Realization.mlx` |
| **S04** | Stability, Feedback & LQR | Lyapunov theory, BIBO stability, pole placement, LQR, Riccati equations, digital control | `S04_Stability_Feedback_LQR.mlx` |

Check [INDEX.md](INDEX.md) for more details.

##  Learning Objectives

By completing this series, you will be able to:

- Represent physical systems in state-space form
- Linearize nonlinear systems around operating points
- Analyze system controllability and observability
- Assess stability using multiple criteria
- Design state feedback controllers (pole placement and LQR)
- Design full and reduced-order observers
- Implement complete observer-based control systems


##  Requirements

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

## Key MATLAB Functions by Season

### S01 - Mathematical Foundations
`eig`, `inv`, `det`, `rank`, `trace`, `norm`, `null`, `orth`, `poly`, `jordan`, `expm`, `linsolve`

### S02 - Modeling & Linearization
`ss`, `tf`, `ss2tf`, `tf2ss`, `c2d`, `d2c`, `step`, `impulse`, `initial`, `lsim`, `jacobian`, `laplace`, `ilaplace`, `ode45`

### S03 - Controllability & Observability
`ctrb`, `obsv`, `rank`, `gram`, `canon`, `ss2ss`, `minreal`, `balreal`

### S04 - Stability & Feedback
`lyap`, `dlyap`, `chol`, `place`, `acker`, `lqr`, `dlqr`, `care`, `dare`, `pole`

## Related Resources

### Recommended Textbooks
- **Modern Control Fundamentals** by A. Khaki Sedigh
- **Modern Control Engineering** by Ogata
- **MATLAB for control system engineers** by Rao V Dukkipati


### MATLAB Documentation
- [Control System Toolbox Documentation](https://www.mathworks.com/help/control/)



##  Support and Feedback

### Getting Help
- Review previous seasons if concepts are unclear
- Check MATLAB documentation (`help function_name`)
- Experiment with simpler examples first
- Use MATLAB's debugging tools (`dbstop`, breakpoints)
- Email me at mehrzadgolabi@gmail.com or @mehrzad_golabi on telegram

## Reporting Issues
If you find errors or have suggestions:
- Document the issue (script name, section, error message)
- Provide MATLAB version and toolbox versions
- Suggest improvements or clarifications
- Open a pull request

## Acknowledgments

This series integrates concepts from:
- Modern control theory literature and Dr. Nouri Manzar's Slides
- MATLAB Control System documentation
- Feedback from teaching assistant colleagues. Special thanks to A. Feizbakhsh and S. Aligholizade for preparing the first draft of this course and providing valuable feedback.
- A. Moradi Amani, Negahi be Karbord-e Narmafzar MATLAB dar Control-e Modarn (نگاهی به کاربرد نرم افزار متلب در کنترل مدرن), appendix to Modarn Control Fundamentals, by A. Khaki Sedigh.
