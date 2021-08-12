# Exam plan

## Plan

### 1. Requirements => ???

### 2. Modelling 

- Explaining the meaning of functions ..  :white_check_mark:
-  Writing functions/ code: :neutral_face:

### 3. Testing

- Provinding statements that achieve x % coverage :white_check_mark:

### 4. Abstract Interpretation

- interval analysis :no_entry_sign:
- designing abstract transformers :neutral_face:

### 5. Pointer Analysis 

- Flow sensitive/ Flow insensitive: Given state, write the corresponding program :neutral_face:
- Given object structure, find points-to sets :white_check_mark:

### 6. Symbolic Execution

- Find what is the required input for ex. loop exactly three times :white_check_mark:
- Proof using symbolic execution, by writing all possible traces fulfilling a certain constraint :no_entry_sign:
- 

## To Remember

### Abstract Interpretation

- Apply standard Interval Analysis to compute sound andiron-trivial invariants at fix-point:

  => per line: write the values each variable can take.

  - at the beginning all are beween minus and plus infinity
  - when given a value : between [value, value]
  - when inside a loop: between [value, infty] => if counter variable and augmenting
  - **Widening:** 

### Pointer Analysis

#### Flow Sensitive Pointer Analysis

-  abstract state S, give a program that results in the given state : if there is an assignement to a field, do it first, then the assignement to the objects themselves.

#### Flow Insensitive Pointer Analysis

- if two labels point to the same object and another one, we cannot distinguish between what they are pointing to, hence cannot assign a field of one to another .

### Symbolic Execution

- Combines testing and static analysis

- Completely automatic, aims to explore as many program executions as possible

- **But**: has false negatives: may miss program executions/ errors

- At any point during execution, SE keeps two formulas : 

  - **Symbolic Store: **$\sigma_s \in SymStore=Var \rightarrow Sym$, with 

    - Var: set of variables as before
    - Sym: set of symbolic values
    - $\sigma_s$: symbolic store

    => Example:$z=x+y$​ will produce $\sigma_s:x\rightarrow x_0, y \rightarrow y_0, z \rightarrow x_0+y_0$. ​

  - **Path Constraint**: records the history of all branches taken so far. 

    - At the start of the analysis the constraint is set to true
    - Evaluation of the conditionals affects the path constraint but not the symbolic store.

  > Example :
  >
  > Let: $\sigma_s:x\rightarrow x_0, y \rightarrow y_0$​, and $pct=x_0 >10$​​ 
  >
  > Let’s evaluate: $if (x>y+1) {5:..}$
  >
  > Then: at label 5 we will get the same symbolic store $\sigma_s$. 
  >
  > But we will get an updated path constraint: $pct=x_0>10 \wedge x_0 > y_0+1$ 
  >
  > 
  >
  > 