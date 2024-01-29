# Optimizing K-Nearest Neighbors Classifier to Predict Customer Churn

## Research Question
Any company that provides a service wants to maintain their existing customer base. Churn is a commonly used term related to customers discontinuing their terms of service. For an ISP a customer who has “churned” has ceased to continue to use the services for which they have subscribed. It is more costly to attain new customers than it is to acquire new ones, so minimizing churn is a crucial aspect of maintaining profitability and providing good service. The Customer Churn data set contains a Churn column that is defined by whether a customer has ended their services within the last month, indicated by a Yes or No value.

The specific products and services offered may play an influential role in a customer’s decision to stop their subscriptions. Such a collection of ISP controlled offerings may serve as input for a predictive model for customer churn. This may beg the question of how a saturated collection of ranges of input variables may be used to formulate such predictions. A logistic classification model dependent on a linear combination of predictor variables could be applied in this situation, but there may exist further subtleties. In a logistic model a predictor variable is ascribed a static coefficient of variability in its contribution to prediticting the likelihood of the outcome.

There may be ranges of a predictor variable which conrtibute more weight on the outcome than others, and there may be more than one of these such sets of ranges. Perhaps the longer a customer experiences an outage on average may steadily impact the likelihood of churn overall, but what if there were a concentration of outage times that have more of an impact than others? Perhaps customers in a mid-to-high range of outage times were more likely to churn than those who experience a more middle of the range outage times. It would seem that, if this were the case, then such ranges would have differing impacts on predicting churn. This is the central notion of the K-Nearest Neighbors (KNN) classifier. Logistic classifiers draw a single smooth decision boundary on its prediction, whereas KNN may divy up several complex boundaries. (Supervised Learning with scikit-learn , Ch. 1, lesson 1). Does such a classifier predict churn better?

## Objective
A logistic predictive model can be applied to make predictions for an outcome with only 2 possible values, but in order to capture the kinds of subtleties mentioned above K-Nearest Neighbors classification may provide more accuracy. Beginning with a set of attributes that describe customer charateristics an initial model can be made. Model performance attributes can be examined to make improvements, which may require certain parameters to be adjusted. The objective is to optimize parameters to obtain a KNN model that most accurately predicts churn. Furthermore, it would be interesting to see if it can outperform logistic classification.

## Data Goals
Using computational quantitative modelling can be used to aid in data driven insight. Churn is a binary valued field, and such modelling can be used to predict the likelihood or probability of a yes or no churn decision occurring. Probability can be calculated but require numerical input. There are several categorical fields in the data set that may be useful to make a prediction. If such fields could be ascribed to numerical values they can be of use.

The data set includes several fields that describe product and service offerings that customers can choose from. Examples of such fields include InternetService, DeviceProtection, StreamingTV, and Contract. These all have categorical values of Yes/No, or a small number of options. There are also some fields that may be descriptive of customer experience, but can be influenced to some extent by the ISP. These type of predictors generally have a continuous range of values. Email describes how many times the ISP has contacted a customer via email. MonthlyCharge is what the customer pays for their services, but can be raised or lowered at the discretion of high level management. If the outcome of predictive modelling shows that churn can be reduced by lowering bills, then perhaps promotions or discount offerings may be worth implementing. Bandwidth_GB_Year is an indicator of level of internet usage that occurs in a customer household, but marketing efforts can be tailored to target low or high level users. Outage_sec_perweek and Yearly_equip_failure can be indluenced by the ISP through increased focus on network maintenance and better device offerings.

Fields that are more descriptive of customer characteristics will not be considered in this treatment. These include demographic features that describe geographic location, personal characterisics, and other opinion driven aspects. Honing in on the features that are more controllable with a detailed model may provide more immediate value towards decision making.

This leaves 19 categorical and numeric features to be explored as explanatory variables in the model. The numerical columns can be further subdivided into discrete and continuous.

Categorical: Contract, PaperlessBilling, Port_modem, Tablet, InternetService, Phone, Multiple, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreaminMovies, PaymentMethod

Discrete numerical: Email, Yearly_equip_failure

Continuous: Bandwidth_GB_Year, MonthlyCharge, Outage_sec_perweek

## Assumptions and Method
The KNN classifier alogirthmically considers how close a certain number of surrounding data points are to each other based on a Euclidean distance measure. Even when all categorical predictor variables are transformed numerically the ranges may differ widely. For this reason it is recommended that all variables be normalized on the same scale. The set of predictive and target features can be split into training and testing sets. A certain percentage of data can be used to create the model, while the rest can be used to measure how well it performs. By comparing predictions to actual values in the test set accuracy measures can be computed. A variation of such an accuracy measure can also be used to determine the best number of neighbors to use.

Tool Benefits and Technique
Python has several libraries containing pre-coded functions that can make model building, parameter optimization and computing probabilities and accuracy metrics very efficient. The sci-kit learn library was used heavily. The following list includes packages and functions used:

sklearn

model_selection

train_test_split - splitting the dataset

GridSearchCV - parameter optimization

neighbors

KNeighborsClassifier - KNN model creation

metrics

confusion_matrix - see below for explanation

classification_report - model performance metrics summary

roc_auc_score - see below for explanation

roc_curve - see below for explanation

preprocessing

StandardScalar - scaling/normalizing data

Pipeline - apply multiple operations on data

linear_model - logistic regression model

pandas - general dataframe handling

Splitting the data allows for computations of True Positive (TP), True Negative (FN), False Positive (FP), and False Negative (FN) outcomes. These 4 values are typically summarized in a confusion matrix, and used to cumpute various model performance metrics.

Model Accuracy is essentially the ratio of the number of correct predictions to total number of predictions.

Accuracy = TP + TN / (TP + TN + FP + FN)

True Positive Rate (TPR) and False Positive Rate (FPR) are also comuputed from these 4 values. A plot of TPR vs. FPR gives an ROC (receiver operating chatacteristic) curve. The area under the curve (auc) provides a measure of how well a variable contributes the prediction; 0 being weakest to 1 being strongest. (Machine Learning Crash Course)

TPR = TP / (TP + FN)

FPR = FP / (FP + TN)

from sklearn.metrics import roc_auc_score

Precision is the proportion of positive instances that were correctly identified. Recall is the proportion of actual positive cases that were correctly predicted. It is clear that a good model will have both high sensitivity and specificity. If these ratios are too low the model may be overfit to the test data.

Precision = TP / (TP + FP)

Recall = TP / (TP + TN)

The F1-score considers both, and is used in a process called grid search cross validation to determine the optimal number of neighbors.

F1 = 2 x (Precision x Recall) / (Precision + Recall) (Analytics Vidhya)

In KNN larger values of K generally leads to a smoother decision boundary, while smaller K has a more complex decision boundary which can lead to overfitting. The .score() function provides a simple means of measuring accuracy of model predictions on a sample measured against actual outcomes in the same sample. (scikit-learn.org). A plot of training and testing set accuracies can be made for varying K's. Such a plot is called a model complexity plot. (Data Camp - Supervised Learning with scikit-learn, Ch. 1) In some cases when K becomes too large the accuracies of training and testing samples diverge from each other, and this is where underfitting occurs. The sweet spot on the model complexity plot occurs when the accuracy measures are closest together.
