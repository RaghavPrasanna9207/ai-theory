# Training Models

## Linear Regression
*   **Concept**: Makes predictions by simply computing a weighted sum of the input features plus a baseline value called a bias (or intercept term).
*   **Training**: Training a model means setting the parameters so that it best describes the data it is trained on while maintaining its ability to generalize.
*   **Cost Function**: To measure how poorly the model is performing, we use Root Mean Squared Error (RMSE). In practice, it is easier to minimize the Mean Squared Error (MSE), which yields the same result.
*   **The Normal Equation**: A closed-form mathematical equation that directly computes the exact parameters needed to minimize the cost function in one go.
    *   **Computational Complexity**: Computing the inverse of the matrix takes between $O(n^{2.4})$ and $O(n^3)$. This means the Normal Equation becomes extremely slow when the number of features grows large (e.g., 100,000).
    *   **Performance**: It scales linearly with regards to the number of training instances. Once trained, making predictions is very fast.

## Gradient Descent (GD)
*   **Concept**: A generic optimization algorithm that iteratively tweaks parameters in order to minimize the cost function.
*   **How it Works**: It starts by filling the parameter vector with random values (random initialization). It then measures the local gradient of the error and takes steps in the opposite direction of the gradient until it reaches zero, which is the minimum.
*   **Learning Rate**: The most important hyperparameter, representing the step size.
    *   If too small, the algorithm takes many steps and is slow to converge.
    *   If too large, it might jump completely across the minimum and end up even higher on the other side.
*   **Cost Function Shape**: The MSE for Linear Regression is a convex function. It looks like a perfectly regular bowl with no irregular holes, ridges, or plateaus, guaranteeing a single global minimum.
*   **Feature Scaling**: If features have different scales, the cost function takes the shape of an elongated bowl. You must ensure all features have a similar scale, or the algorithm will take much longer to converge.

### Types of Gradient Descent
1.  **Batch Gradient Descent**: Uses the complete training set to compute gradients at every single step.
    *   **Pros/Cons**: Extremely slow with large datasets, but scales perfectly with a large number of features.
    *   **Tolerance**: To decide when to stop training, you set a tiny number called tolerance. Whenever the norm of the gradient vector becomes less than this tolerance, training is interrupted.
2.  **Stochastic Gradient Descent (SGD)**: Picks a single random instance in the training set at every step and computes gradients based only on that instance.
    *   **Pros/Cons**: Blazing fast and requires very little memory (supports out-of-core algorithms).
    *   **Behavior**: Due to its random nature, it bounces around and never truly settles at the optimal minimum. However, this bouncing helps it jump out of local minima in irregular cost functions.
    *   **Simulated Annealing**: To help the algorithm eventually settle, the learning rate is gradually reduced over time using a function called a learning schedule.
3.  **Mini-Batch Gradient Descent**: Computes gradients based on small, random sets of instances called mini-batches.
    *   **Pros/Cons**: Gets a significant performance boost from hardware optimization of matrix operations (especially on GPUs). It is less erratic than SGD but slightly more susceptible to getting stuck in local minima.

## Polynomial Regression
*   **Concept**: Fits non-linear data to a linear model. It accomplishes this by adding powers of each feature as new features, and then training a standard linear model on this extended, higher-dimensional set of features.

## Evaluating Models: Learning Curves
Learning curves plot a model's performance on both the training set and validation set as a function of the training set size. They are a primary tool for diagnosing fit.

*   **Underfitting**: The model is too simple to capture the underlying pattern. The error on the training data goes up and reaches a high plateau. The validation error also reaches a high plateau, remaining very close to the training curve. Adding more training examples will not help; you must use a more complex model or engineer better features.
*   **Overfitting**: The model memorizes the training data but fails to generalize. The training error is remarkably low, but the validation error is significantly higher, leaving a large, noticeable gap between the two curves. Adding more training examples will help reduce overfitting and bring the curves closer together.

## Regularization
Regularization reduces a model's risk of overfitting by constraining its weights, effectively reducing its degrees of freedom. It is essential to scale your data before performing regularization, as these models are highly sensitive to the scale of the input features.

*   **Ridge Regression (Tikhonov Regularization)**: Regularizes the model by adding an $L_2$ norm penalty to the cost function during training. This forces the algorithm to fit the data while keeping all feature weights as small as possible. If the regularization parameter (alpha) is too large, the weights approach zero, and the model turns into a flat line passing through the data's mean.
*   **Lasso Regression**: Regularizes the model by adding an $L_1$ norm penalty. A defining characteristic of Lasso is that it tends to completely eliminate the weights of the least important features (setting them to zero). It performs automatic feature selection and outputs a sparse model.
*   **Elastic Net**: A middle ground that mixes the Ridge and Lasso penalties via a customizable ratio.
    *   **When to use which**: Ridge is a great default starting point. If you suspect that only a small subset of features is actually useful, use Lasso or Elastic Net. Elastic Net is generally preferred over pure Lasso because Lasso can behave erratically when the number of features exceeds the number of training instances, or when several features are strongly correlated.
*   **Early Stopping**: A completely different approach to regularization. You simply monitor the validation error during training and immediately stop the process as soon as the validation error reaches its minimum.

## Logistic Regression
*   **Concept**: A binary classifier used to predict the probability that an instance belongs to a particular class (the positive class).
*   **How it Works**: Just like Linear Regression, it computes a weighted sum of the input features. However, it outputs the "logistic" of that result by passing it through a sigmoid function, which squashes the value into a probability between 0 and 1.
*   **Decision Boundary**: If the estimated probability is greater than 50%, the model predicts the instance belongs to the positive class. Otherwise, it predicts the negative class.
*   **Cost Function**: Uses a cost function called Log Loss. There is no closed-form solution to calculate the optimal weights, but the function is convex, making it perfectly suited for Gradient Descent.
*   **Hyperparameter C**: In Scikit-Learn, the regularization strength of a Logistic Regression model is controlled by the hyperparameter `C` (which is the inverse of alpha). A higher value of `C` means the model is less regularized.

## Softmax Regression (Multinomial Logistic Regression)
*   **Concept**: Generalizes Logistic Regression to support multiple mutually exclusive classes directly, without needing to train multiple binary classifiers.
*   **How it Works**: When given an instance, the model computes a specific score for every single class. It then applies the Softmax function (normalized exponential) to these scores to estimate the probability of each class. The model predicts the class with the highest probability.
*   **Cost Function**: Uses the Cross-Entropy loss function to measure how wrong the estimated probabilities are compared to the target classes.
*   **Limitation**: The classifier only predicts one single class at a time. Therefore, it cannot be used for multi-output recognition, such as identifying multiple different people in the exact same photograph.
*
## Formulas

### Linear Regression
**Linear Regression Model (Vectorized):**
$$\hat{y} = \theta^T x$$

**Mean Squared Error (MSE) Cost Function:**
$$MSE(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)})^2$$

**The Normal Equation:**
$$\hat{\theta} = (X^T X)^{-1} X^T y$$

### Gradient Descent
**Partial Derivative of the Cost Function:**
$$\frac{\partial}{\partial \theta_j} MSE(\theta) = \frac{2}{m} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)}) x_j^{(i)}$$

**Gradient Vector of the Cost Function:**
$$\nabla_\theta MSE(\theta) = \frac{2}{m} X^T (X \theta - y)$$

**Gradient Descent Step (where $\eta$ is the learning rate):**
$$\theta^{(\text{next step})} = \theta - \eta \nabla_\theta MSE(\theta)$$

### Regularization
**Ridge Regression Cost Function:**
$$J(\theta) = MSE(\theta) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2$$

**Lasso Regression Cost Function:**
$$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|$$

**Elastic Net Cost Function (where $r$ is the mix ratio):**
$$J(\theta) = MSE(\theta) + r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1 - r}{2} \alpha \sum_{i=1}^{n} \theta_i^2$$

### Logistic Regression
**Sigmoid Function:**
$$\sigma(t) = \frac{1}{1 + e^{-t}}$$

**Estimated Probability:**
$$\hat{p} = \sigma(\theta^T x)$$

**Log Loss Cost Function (for the whole training set):**
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

### Softmax Regression
**Softmax Score for Class $k$:**
$$s_k(x) = (\theta^{(k)})^T x$$

**Softmax Function (Probability of Class $k$):**
$$\hat{p}_k = \frac{\exp(s_k(x))}{\sum_{j=1}^{K} \exp(s_j(x))}$$

**Cross-Entropy Cost Function:**
$$J(\Theta) = - \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{p}_k^{(i)})$$