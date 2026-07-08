# Classification

## Training a binary classifier
1. Classifies between two classes.

## Performance measures

1. Measure accuracy using cross-validation. Accuracy is generally not a good performance measure for classification, especially in skewed datasets. A *confusion matrix* is much better.
2. Confusion matrices - use *cross_val_predict()* to implement these. Each row in a CM is an actual class, while each column is a predicted class. Precision = TP / TP + FP, Recall = TP / TP + FN. Precision is, out of all responses predicted positive, how many are true. Recall is, out of all actual positive responses, how many are predicted correctly. *F1 Score* is the harmonic mean of both. Prioritise precision if accuracy is more important and true values getting flagged false is not a problem, prioritise recall if false alerts are fine.
3. *Precision/Recall Trade-Off* - Increasing one reduces the other. There's also a *threshold*, a cutoff value used to convert a model's probability into a label. As the value increases, precision increases and recall decreases.
4. The ROC (Receiver Operating Characteristic) curve plots the *true positive rate(recall/sensitivity)* against the *false positive rate(fall-out)*. FPR = 1 - *TNR(Specificity)*. Higher the TPR, more the FPR produced. Area under the curve - *AUC* - can be used to compare classifiers. A perfect classifier has AUC = 1, while a truly random one has 0.5.
5. Use the PR curve if you care more about the false positives than the false negatives, use ROC otherwise.
6. Multiclass classifiers - can distinguish between more than two classes. Multiple binary classifiers can be put together to become a multiclass classifier, where one classifier's score is chosen. this is OvR or OvA (One versus Rest/All). One versus One (OvO) uses classifiers for every pair of digits.
7. Data augmentation - slightly changing the data so that the model can make better predictions.
8. Multilabel classification system - Outputs multiple binary labels. To evaluate, use F1 score for each induvidual label and compute the average. Use *ClassifierChain* to feed the model the input features and all previous predictions.
9. Multioutput classification - A generalisation of multilabel where each label can have more than two possible values.