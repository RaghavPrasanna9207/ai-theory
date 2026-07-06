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
14. Handle text and categorical values. They can be converted to numbers using *OrdinalEncoder*. *OneHotEncoding* gives it binary values for each column, making it easier to use. Sklearn also has some functions which make it easy for debugging and ensuring consistency of columns.
15. Scale features and transform. Model will not work well if attributes are of different scales. This can be done by *Min-Max Scaling/Normalisation* or *Standardization*. MMS can set it from 0 - 1 or -1 - 1. Standardization uses mean and SD, and is less likely to get affected by outliers. Make sure all graphs are bell-curved. Apply log or sqaure root to ensure that. *Bucketizing* gives each attribute a percentile value, giving a uniform distrubution.
16. For multimodal features, either one hot encode the buckets or add another feature representing similarity, which is usually computed using *Radial Basis Function(RBF)*. RBF measures how similar something is to a reference point.
17. Create custom sklearn transformers if needed.
18. Use *Pipelines* for a sequence of transformations. Use make_pipeline to do it easily. Everything in the pipeline should have a fit_transform() method, except for the last one, which can be anything. *Column Transformers* help select specific columns and transform them using pipelines. One pipeline can be used for all the preprocessing and cleaning.
19. Train and evaluate on a training set.
20. For better evaluation, use *Cross-Validation*. If the training set is seperated into two sets - training and validation, use the latter for validation. *k-fold cross validation* can be used for the same. If k is 10, the set is seperated into ten parts, and each part is the validation set once, so we get 10 results. Use *cross_val_score()* for results.
21. Fine-tune the model. Use *GridSearchCV* to find the best set of hyperparameters. Use *RandomizedSearchCV* for bigger datasets.
22. Analyse the model and drop features which don't have much importance. Use *feature_importances* for that. *sklearn.feature_selection.SelectFromModel* does that for you. Also check model fairness, make sure model is not biased and works well everywhere.
23. Evaluate your system on the test set. Performance might be worse, but do not change hyperparameters because of that.
24. Create concise reports, visualise data, tailor message to the audience. Provide impactful and easy-to-remember statements. Highlight what you have learned, what worked and what did not, what assumptions were made, and what your system’s limitations are.
25. Your results should be reproducible (as much as possible): make the code accessible to your team (e.g., via GitHub), add a structured README file to guide a technical person through the installation steps. Provide clear notebooks (e.g., Jupyter) with code, explanations, and results, writing clean, well-commented code. Define a requirements.txt or environment.yml file containing all the required libraries along with their precise versions (or create a Docker image). Set seeds for random generators, and remove any other source of variability.
26. Launch, monitor and maintain the system. Use *joblib* to import and export the model. Collect fresh data regularly and label it, write scripts to automatically train and fine-tune the model, write another script to evaluate both new and old models and deploy to production if it has not decreased.