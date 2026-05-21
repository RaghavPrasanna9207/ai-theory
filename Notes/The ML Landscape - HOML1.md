## How do you use ML?

1. Study the Problem
2. Feed data → Train model → Solution
3. Inspect solution → Understand problem better
4. Repeat if needed

## Types of Systems

- **Supervised Learning** - Training set fed to the algo includes the desired solns
- Classification, regression,
- **Unsupervised Learning** - Training data is unlabelled
- Clustering algos, visualisation algos,dimensionality reduction, feature extraction, anomaly detection, association rule learning
- **Semi Supervised** - Few labeled and few unlabelled. Often has tasks comprising of both supervised and unsupervised.
- Example: Google Photos. Using supervised the model says this face = this person, unsupervised recognizes this face shows up in these pictures.
- **Self Supervised** - Generating fully labelled datasets from fully unlabelled ones.
- **Reinforcement Learning -** The system (called the agent) can observe the envi, select and perform actions and get rewards and penalties in return.
- The action it chooses in a situation is called the *policy* . Iteration repeats till optimal policy found.
1. Observe
2. Selection action using policy
3. Action
4. Reward/Pen
5. Update policy
6. repeat until optimal

## Batch vs Online Learning

- **BATCH -** System is trained using all avlbl data.
- It is done offline. Learning stops when launched.
- Model’s performance also tends to decay over time. This is *Data Drift.*
- **ONLINE** - System is trained by feeding it data sequentially, either induvidually or in small groups(*mini batches*)
- Each step is fast and cheap so that the system can learn about new data quickly.
- Train model, evaluate, launch, run and learn.
- How fast the model adapts to data - *learning rate* - is an important parameter.
- A high learning rate not only learns fast, but also forgets fast. This is *catastrophic forgetting*.

## Instance Based vs Model Based Learning

- **INSTANCE BASED** - System learns the examples and generalizes to new cases by using a similarity measure.
- Does well with small datasets. Does not scale well.
- Is a bit slow, doesnt work well with high dimensional data(like images).
- **MODEL BASED** - Typical ML workflow

## Main Challenges

- Insufficient quantity of training data
- Nonrepresentative training data
May cause *sampling bias* - samples being large/small in number causes non-rep.
- Poor quality of data
- Some outlier may be discarded
- Some instances missin gfeatures can be ignored or filled in
- Irrelevant Features
- This is where *feature selection* and *feature extraction* come into play.
- Overfitting
- Can be overcome by *regularization* (making the model simpler). Amount of regularisation can be controlled by a *hyperparameter* (parameter of a learning algorithm)
- Underfitting
- Deployment issues

## Testing and Validating

- Data split into *training set* and *test set*. Error rate on new cases is called the *generalisation error,* an estimate of this we get from the test set.
- If training error is low (few mistakes) but gen error high, overfitting.
- Usually, **80/20 training/test**.
- *Holdout Validation* - when a part of the training set is held out (*validation set* ) to evaluate multiple models and select one.
- You train multiple models with various hyperparameters on the training set and you select the model that performs best on the validation set. Then, you evaluate this on the test set.
- We can use *Cross Validation*  to improve this even more. Here we use many small validation sets, where each model is evaluated once per validation set after it is trained on the rest of the data. Average out the evaluations and you get a much more accurate measure of its performance. This solves the problem of the validation set being too big or small, but has its own problem - training time getting increased.
- There is also a *Train-Dev Set* (before dev after train) to distinguish between errors caused by overfitting and errors caused by data msimatch.
- Both validation and test sets must be as representative as possible of the data you expect to use in production. (you can shuffle and put half in this, half in that)

## Questions and Answers

### 1. How would you define machine learning?

Machine Learning is the science (and art) of programming computers so they can **learn from data**. Instead of hard-coding rules, the system analyzes examples to find patterns and build its own logic.

- **Formal Definition:** A program learns from experience $E$ with respect to task $T$ and performance measure $P$, if its performance on $T$, as measured by $P$, improves with experience $E$.

### 2. Can you name four types of applications where it shines?

1. **Complex problems with no existing solution:** e.g., Speech recognition.
2. **Fluctuating environments:** e.g., Spam detection (adapts to new spam tactics).
3. **Getting insights from large data (Data Mining):** e.g., Analyzing purchase data for hidden segments.
4. **Replacing long lists of hand-tuned rules:** e.g., Image classification.

### 3. What is a labeled training set?

It is a dataset containing the **inputs** (features) along with the **desired solution** (labels) for each instance.

- *Example:* In a housing price project, the features are "size" and "location," and the label is the "price."

### 4. What are the two most common supervised tasks?

1. **Classification:** Predicting a discrete class/category (e.g., Spam vs. Ham).
2. **Regression:** Predicting a continuous numerical value (e.g., Price of a house).

### 5. Can you name four common unsupervised tasks?

1. **Clustering:** Grouping similar data points together (e.g., K-Means).
2. **Visualization:** Projecting high-dimensional data into 2D/3D to view patterns (e.g., t-SNE).
3. **Dimensionality Reduction:** Simplifying data without losing information (e.g., PCA).
4. **Association Rule Learning:** Digging into large transactional data to find interesting relations (e.g., "People who buy beer also buy diapers").

### 6. What type of algorithm would you use to allow a robot to walk in various unknown terrains?

**Reinforcement Learning.**
The robot (agent) observes the environment, takes an action (steps), and receives a **reward** (staying upright) or a **penalty** (falling). It learns a policy to maximize rewards over time.

### 7. What type of algorithm would you use to segment your customers into multiple groups?

**Clustering** (Unsupervised Learning).
Since you likely don't know the groups in advance (no labels), you let the algorithm find natural clusters based on purchasing behavior or demographics.

### 8. Would you frame the problem of spam detection as a supervised or unsupervised learning problem?

**Supervised Learning.**
You typically train the model with a dataset of emails already flagged as "Spam" or "Ham" (labels).

### 9. What is an online learning system?

A system that learns incrementally as data comes in, rather than all at once. It is capable of adapting to changing data rapidly and autonomously.

- *Use case:* Stock price prediction or systems with limited storage.

### 10. What is out-of-core learning?

**Out-of-core learning** is a technique used when a dataset is too massive to fit into your computer's main memory (RAM).

- **How it works:** The algorithm loads the data in small chunks (mini-batches), trains on that chunk, and then discards it to load the next one.
- **Key Fact:** It is typically implemented using **Online Learning** algorithms.

### 11. What type of algorithm relies on a similarity measure to make predictions?

**Instance-based Learning** algorithms (e.g., K-Nearest Neighbors).

- Instead of learning a general rule or formula, the system memorizes the training examples.
- To predict a label for a new instance, it compares the new instance to the stored examples using a **similarity measure** and adopts the label of the most similar ones.

### 12. What is the difference between a model parameter and a model hyperparameter?

- **Model Parameter:** A configuration variable that is internal to the model and whose value can be estimated from data. The model learns these *on its own* during training.
    - *Example:* The weights ($w$) and biases ($b$) in a neural network or linear regression.
- **Hyperparameter:** A configuration that is external to the model and whose value cannot be estimated from data. You (the human) must set these *before* training begins.
    - *Example:* Learning rate, number of clusters ($k$), or the depth of a decision tree.

### 13. What do model-based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

- **Search for:** The optimal values for the model parameters (like weights) that enable the model to generalize well to new instances.
- **Strategy:** They minimize a **Cost Function** (which measures how bad the model is, like Mean Squared Error) or maximize a **Utility Function** (which measures how good it is). This is often done using an optimization algorithm like **Gradient Descent**.
- **Predictions:** Once the parameters are set, the model plugs the new instance's features into its prediction function to calculate the output.

### 14. Can you name four of the main challenges in machine learning?

1. **Insufficient Quantity of Training Data:** Complex problems require thousands or millions of examples.
2. **Non-representative Training Data:** If the data doesn't represent the real world (sampling bias), predictions will be skewed.
3. **Poor Quality Data:** Errors, outliers, and noise make it hard to detect patterns.
4. **Overfitting:** The model is too complex and memorizes noise rather than the signal.

### 15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?

**What is happening:** The model is **Overfitting**.

**Three Solutions:**

1. **Get more training data:** Helps the model see general patterns.
2. **Simplify the model:** Select a simpler algorithm, reduce parameters/layers, or reduce features.
3. **Regularization:** Apply constraints (like L1/L2) to keep weights small.

### 16. What is a test set, and why would you want to use it?

- **What:** A subset of the dataset (typically 20%) that is set aside and *never* used during training or tuning.
- **Why:** It is used to estimate the **generalization error** of the final model. It acts as a realistic simulation of how the model will perform in production on completely unseen data.

### 17. What is the purpose of a validation set?

The validation set (or development/dev set) is used to **compare models and tune hyperparameters**.

- You train multiple models with different hyperparameters on the *Training Set*.
- You evaluate them on the *Validation Set* to see which one performs best.
- You select the winner and do a final check on the *Test Set*.

### 18. What is the train-dev set, when do you need it, and how do you use it?

- **What:** A specific subset of the *training data* that is held out (not trained on).
- **When:** You need it when there is a risk of **Data Mismatch** (e.g., your training data comes from the web, but your production data comes from mobile cameras).
- **How to use:**
    - Train on the *Training Set*.
    - Evaluate on the *Train-Dev Set*.
    - If it performs poorly on *Train-Dev*, you have an **Overfitting** problem (variance).
    - If it performs well on *Train-Dev* but poorly on the *Validation Set*, you have a **Data Mismatch** problem.

### 19. What can go wrong if you tune hyperparameters using the test set?

If you use the test set to tune hyperparameters, you will **overfit to the test set**.

- The test set is supposed to be an unbiased evaluator. If you tweak the model until it gets a perfect score on the test set, that score is no longer a valid estimate of general performance. The model has "seen" the test questions and adjusted specifically for them, leading to poor performance in the real world.