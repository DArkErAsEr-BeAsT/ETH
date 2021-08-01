### Regression

- for Ordinary Least Squares, what increases the loss :
  - setting the bias term to zero or not fitting a bias term
  - doing dimensionality reduction
- The ridge estimator has larger bias and smaller variance than the OLS estimator
- Ridge regression:
  - the norm of the optimal weight vector is a  monotonically decreasing function of $\lambda$
  - the objective function has a unique optimiser
- Lasso Regression selects a subset of the input features.

- Greedy Forward Selection is faster than Backwards selection if only a few features are relevant
- Computational Complexity of computing the closed form solution of linear regression : $O(nd^2)$

- Correct expression for computing $\Delta L(w) = 2X^TXw-2X^Ty$ ( when minimising the linear regression equation using gradient descent) + Computational Complexity of computing it : $O(dn)$

### Kernels 

- Properties:

  - k is symmetric : $k(x,x’)=k(x’,x)$
  - k is an inner product in a suitable space

  

### Support Vector Machines

​	