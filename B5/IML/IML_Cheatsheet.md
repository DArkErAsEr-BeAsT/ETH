### General Stuff

- Difference between concave and convex:
  - Concave : that curves inwards
  - Convex: that curves outwards (exterior)

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

- F1-Score: $2* \frac {precision * recall} {precision+recall}$

- ROC Curve: increasing 

### Dimension Reduction & Clustering

- k-means clustering: 
  - seeks cluster centers and assignments to minimise the within cluster sum of squares
  - good when clusters are separable, spherical and the same size
  - can be kernellized

- Lloyd's Algorithm:
  - it never returns to a particular solution after having previously changed to a different solution
  - using specialised initialization schemes can improve runtime and the quality of the solutions

- How to select the number of cluster centers k:
  - by using heuristics like the elbow method that identifies diminishing returns from increasing k
  - by using an information criterion that regularizes solutions  to favor simpler models with lower k
- PCA:
  - unsupervised learning algorithm
  - PCA can be kernelized
- First Principal of PCA
  - orthogonal to all other PC's
  - corresponds to a line that minimizes the sum of the squares of the distances of the sample points from that line.

### Neural Networks

- Backpropagation for computing gradients when training neural networks:
  - can be applied to compute gradients for NN for unsupervised learning
  - it is based on the chain rule for differentiation
- Vanishing Gradient Problem:
  - nn with ReLU activation are less prone to suffer from it
  - solution: batch normalization can sometimes alleviate the problem
- Adding an additional hidden layer with a non-linear activation function to the network will result in a lower training error
- Methods to mitigate overfitting:
  - Early Stopping
  - Dropout
  - Weight Decay
  - Batch Normalisation
- CNN: Pooling layers reduce the spatial resolution of the image

