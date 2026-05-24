# End to End ML Project

*Note: This project was done by me around a year back. Due to that, the project won't be repeated and this repo will have the file straight from the book. This MD file contains theory and serves as a summary of the chapter.*

## Frame the Problem

1. Determine the objective of the model - this helps decide the algorithms, success metrics and the way the problem is framed.
2. Find out what the current situation is. The current situation will often give you a reference for performance, as well as insights on how to solve the problem.
3. Determine what kind of *supervision* is needed. Determine if it's a *classification/regression/other* task.
4. Select a performance measure. **RMSE** is a very common one. It gives an idea of how much error the system typically makes in its predictions, with a higher weight given to large errors.

$$
\text{RMSE}(\mathbf{X}, y, h)=
\sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
\left(
h\left(\mathbf{x}^{(i)}\right)-y^{(i)}
\right)^2
}
$$

$\mathbf{X}$
→ Feature matrix (all input data)
$\mathbf{x}^{(i)}$
→ Feature vector of the i-th training example
Example: [area, bedrooms, age]
$y$
→ Actual target values / labels
$y^{(i)}$
→ True output for the i-th example
$h$
→ Hypothesis / prediction function (the ML model)
$h(\mathbf{x}^{(i)})$
→ Model’s prediction for the i-th example
$m$
→ Number of training examples
$\sum_{i=1}^{m}$
→ Sum over all training examples
$\left(h(\mathbf{x}^{(i)}) - y^{(i)}\right)^2$
→ Squared prediction error for one example

**MAE** is also used. RMSE is sensitive to outliers and MAE can be used if there is an abundance of them.

$$
\text{MAE}(\mathbf{X}, y, h)=
\frac{1}{m}
\sum_{i=1}^{m}
\left|
h\left(\mathbf{x}^{(i)}\right)-y^{(i)}
\right|
$$

5. Check assumptions made by anyone working on the system, or people who work on the output of the system.
6. Get the data. Download it - preferably using a function. *From here, refer notebook for a more detailed guide.*
7. Take a look at the data - use *head()* which shows the top 5 rows, *info()* which gives a description of the data column-wise, & *describe()* which shows a summary of the numerical attributes. Plotting a histogram for each.
8. Create a **Test Set**. Pick 20% (Or lesser, if the dataset is larger) of the dataset and set it aside to test it. *Data Snooping Bias* - A pattern noticed in the test set but has nothing to do with the whole set. Make sure test set is a representation of the whole set - **Stratified Sampling**.
9. Explore and visualise the data - Create an **Exploration Set** for the same. If the data set is small, a copy of the training set can be used.
10. Look and compute correlations using *Pearson's*. Use the values and compute graphs to check it.
11. Experiment with attribute combinations. Make sure combinations actually amount to something and are not just weighted sums of existing features.
12. Prepare the data for algorithms. Revert to a clean training set for the same. Remove the value which is to be predicted and extract it as a column.
13. Clean the data. Handle empty values using *drop(), dropna() and fillna()*. Missing values can be filled with median, mean, most common or constant values.