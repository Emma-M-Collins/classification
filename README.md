# Classification: Random Forest and Logistic Regression on Colinear Data

* There are multiple ways to predict binary (no/yes or 0/1 data), logistic regression and random forests are two very popular techniques
* Use the [Sonar dataset](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29), from University of California, Irvine's [Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
* The Sonar dataset has 60 different measurements and angles of sonar (the predictors) used to predict whether a buried object is a rock or a mine.  There are 208 total observations, with 111 Mine responses and 97 Rock responses

## Method Comparisons
We will use the following statistics to compare logistic regression to random forest
* **Accuracy**: the proportion of correct prediction to all predictions
* **Precision**: the proportion of mines correctly identified to all mines identified (correctly and incorrectly)
* **True Positive Rate (TPR)**: the proportion of mines correctly identified to all known mines
* **False Positive Rate (FPR)**: the proportion of rocks incorrectly identified to all known rocks
* **False Negative Rate (FNR)**: the proportion of mines incorrectly identified as a rock to all known mines
* **ROC curve/AUC**: a diagnostic plot, shows how TPR and FPR vary as the decision threshold changes

## Results
Data split into 75% training data and 25% test data.

**Logistic Regression**
* Highly colinear data, removed 17 predictors (out of 60 total) before building the model
* Final model only has 12 predictors 

**Random Forest**
* 2000 decision trees
* 7 predictors at each split

Training Data
| Method             | Accuracy | Precision |  TPR |  FPR |  FNR |
| ------------------ | --------:| ---------:| ----:| ----:| ----:|
|Logistic Regression |     0.843|      0.178| 0.181| 0.824| 0.821|
|Random Forest       |         1|          1|     1|     0|     0|

<br>

Test Data
| Method             | Accuracy | Precision |  TPR |  FPR |  FNR |
| ------------------ | --------:| ---------:| ----:| ----:| ----:|
|Logistic Regression |     0.615|      0.625| 0.714| 0.333| 0.429|
|Random Forest       |     0.808|      0.750| 0.964| 0.042| 0.321|

From the tables, random forest out-performs logistic regression in every statistic in the training and test data, although the perfect numbers in the training data for random forest indicates overfitting.

<br>

ROC Curves | & AUC
--- | ---
![](/LRROC.png) | ![](/RFROC.png)

Again, the curves shows random forest outperforming logistic regression.  (AUC = 1 is perfect prediction)

## Conclusions
* Random Forest outperforms logistic regression in every facet
* Even after overfitting to the training data, the random forest has better accuracy and prediction 
* Logistic regression had a massive problem with colinearity, unstable results
* ROC curves also indicates random forest predicts much better, the high AUC indicates the method has high seperability

<br>

**Keep in mind these conclusions only apply to the Sonar dataset!**
* If a data sample meets the assumptions for logistic regression, it will often outperform
* **But**, the Sonar dataset is not one of those datasets

### Random Forest Wins!
