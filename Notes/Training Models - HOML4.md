# Training Models

# Linear Model
- a linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term.

## 1. Simple Linear Regression

\[
\hat{y} = \theta_0 + \theta_1 x
\]

Example from the book:

\[
\text{life\_satisfaction} = \theta_0 + \theta_1 \times GDP\_per\_capita
\]

Where:

- \(\hat{y}\) = predicted value
- \(\theta_0\) = bias (intercept)
- \(\theta_1\) = weight (slope)
- \(x\) = input feature

---

## 2. Multiple Linear Regression

\[
\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n
\]

Where:

- \(n\) = number of features
- \(x_i\) = \(i^{th}\) feature
- \(\theta_i\) = weight for the \(i^{th}\) feature

---

## 3. Vectorized Linear Regression

### Hypothesis Function

\[
\hat{y} = h_{\theta}(\mathbf{x}) = \theta \cdot \mathbf{x}
\]

or equivalently

\[
\hat{y} = \theta^T \mathbf{x}
\]

Where:

- \(\theta\) = parameter vector

\[
\theta =
\begin{bmatrix}
\theta_0\\
\theta_1\\
\vdots\\
\theta_n
\end{bmatrix}
\]

- Feature vector

\[
\mathbf{x} =
\begin{bmatrix}
1\\
x_1\\
\vdots\\
x_n
\end{bmatrix}
\]

The leading 1 accounts for the bias term.

---

## 4. Dot Product Expansion

\[
\theta^T\mathbf{x}
=
\theta_0
+
\theta_1x_1
+
\theta_2x_2
+\cdots+
\theta_nx_n
\]

---

## 5. Mean Squared Error (MSE) Cost Function

For a training set of size \(m\):

\[
\text{MSE}(\mathbf{X},h_\theta)
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
\theta^T\mathbf{x}^{(i)}
-
y^{(i)}
\right)^2
\]

Often written more simply as:

\[
\text{MSE}(\theta)
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
h_\theta(\mathbf{x}^{(i)})
-
y^{(i)}
\right)^2
\]

Where:

- \(m\) = number of training examples
- \(\mathbf{x}^{(i)}\) = feature vector of the \(i^{th}\) training example
- \(y^{(i)}\) = actual target value
- \(h_\theta(\mathbf{x}^{(i)})\) = predicted value

---

## Summary

### Prediction

\[
\boxed{\hat{y}=\theta_0+\theta_1x}
\]

\[
\boxed{\hat{y}=\theta_0+\theta_1x_1+\cdots+\theta_nx_n}
\]

\[
\boxed{\hat{y}=h_\theta(\mathbf{x})=\theta^T\mathbf{x}}
\]

### Cost Function

\[
\boxed{
\text{MSE}(\theta)
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
h_\theta(\mathbf{x}^{(i)})
-
y^{(i)}
\right)^2
}
\]