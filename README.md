
![CreditRisk](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/creditrisk.jpeg)
# Predicting Credit Risk Employing Supervised Machine Learning Models

## Brief Introduction -
Lending services companies allow individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. Data on previous loan applications (both Approved and Rejected) was available. This data can be used to build Machine Learning Models which predicts the credit_risk for approving future loans. 

The archieved dataset available for this project has information on loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts,	derogatory_marks, total_debt, and loan_status. This a binary classification ML problem with loan_status either likely to be 0 (not approved) or 1 (approved).

## Aims and Objective -
The loan_status of the applications will be predicted employing both Logistic Regression and Random Forest Classifier.

      (1) Performance of the models will be asessed with or without a preprocessing step included.
     
      (2) Main performance metrics of ML models such as Recall, Precision, Accuracy, and f1-score will be determined and compared.
  
      (3) Any fine tuning that is needed to improve the performace of the models will be carried out using Feature Selection & Hyperparameter Tuning.
  
## Specific Libraries and modules employed - 
      
      Scikit-Learn ML Library- 
  
      (a) Linear model - LogisticRegression, 
      
      (b) Ensemble - RandomForestClassifier,
      
      (c) model_selection - train_test_split & GridSearchCV,
      
      (d) Preprocessing - StandardScaler,
      
      (e) Metrics - classification_report, precision_score, recall_score,
      
      (f) Feature selction - SelectFromModel

## Project Performance -
## Linear Regression with or without Preprocessing -
![LR](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/new_LR.png)


Linear Regression model performed well with an accuracy scores of 0.9921. However, with the scaling of the data it improved even further to 0.9941. Inclusion of the preprocessing step also improved Recall and f1-scores' scores of the model.

If this was a model with a score of say., 0.7823, then there is a high likelihood of this score increasing further, with a preprocessing step included (such as MinMaxScaler or StandardScaler).

The reasons for this improvement are mainly due to - Linear Regression relying on the linear relationship between input features and output variables; any feature variables whose values are not scaled might not be able to fit well in the model and thus penalize the model.

## Random Forest Classifier with or without Preprocessing -
![RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/RFC.png)

Similar to Linear Regression, Random Forest Classifier also performed really well; with an accuracy score of over 0.99.

Unlike Linear Regression, Random Forest Classifier donot rely on the linear relationship of feature and output variable values, and therefore there was little to no improvement in the performance of this model with a scaling step included.

## Feature Selection and evaluation of Random Forest Classifier
![FS-RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Feature%20Selection%20-%20scores.png)

While developing a ML model, it is essential that features that are included in the model, contribute to the overall performance of the model and its accurate prediction. 

Eventhough, these model (demonstarted above) have already performed well with an accuracy score nearly perfect; any further refinement of this model could potentially improve Precision, Recall, f1-scores. With this in mind, feature imortance was attempted using SelectFromModel. Among the input features, interest_rate, borrower_income, and debt_to_income were identified to contribute the most to the overall predictability of the model.

Further refinement of the model was carried out using these three selected features. This didnot improve the score of an already well performing model. However, this refinement process is of utmost importance when a model is faced with hundreds of features, many of which do not contribute to the overall performance and generate only noise for the model.  

## Hyperparameter Tuning and comparison between the two models (code in Bonus file)
![LR v/s RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Hyperparameter.png)

Having recognized that there is still room for improvement with Precision, Recall, f1-scores, hyperparameter tuning was carried out using GridSearchCV. This improved the Recall and f1-score to near perfect scores. 

If this was a model with an accuracy score much lower than 0.99, then there is a high likelihood of this tuning step improving even the accuracy score.

## Summary and conclusions -

        (1) Linear regression model with Preprocessing steps (StandardScaler, and GridsearchCV) and Random Forest Classifier 
            with Feature Selection were the top performing models that were developed, as a part of this Machine Learning Project.

        (2) These two models had near perfect score for Recall, f1-score and Accuracy.

        (3) These models were able to predict credit risk of current loans with 0.99 accuracy.
            

![Credit](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Credit.jpeg)

## Limitations and Discussion -

        (1) The dataset used for this modeling was imbalanced (the ratio being 1:30; 2500 for Approved to 75036 Rejected). 
            Hence, the distribution of output classes was not equal. 
    
        (2) When working with an imbalanced classification problem such as this one, the minority class is typically of the most interest. 

        (3) In order to rebalance the class districbution in the dataset, random resampling technique can be used. 
            The models are already performing well and therefore random resampling was not attempted. 
            However, closer monitoring of the model performance on newer data should be used to assess if random resampling is necessary.
            
## Predictions:

The written analysis did not contain a prediction on which model will perform better. 
In general, logistic regression performs better when the number of noise variables is less than or equal to the number of explanatory variables and random forest has a higher true and false positive rate as the number of explanatory variables increases in a dataset.

In this article, we discuss Logistic Regression and Random Forest Classifier.

Logistic Regression is used widely for solving Industry scale problems as it’s easy to implement and it generally doesn't give you discrete output and gives Probability associated with each output. The logistic regression algorithm is robust to a small noise in the data and is not particularly affected by small cases of multi-collinearity. We can use many metrics for evaluation but let’s consider Recall for True Positive and F1 score (Harmonic Mean for Recall and Precision)

Random Forest Classifier is more of Accuracy focused algorithm and is best till it’s used with proper fit, else it gets overfat quickly. Random selection in individual decision trees of RFC can capture more complex feature patterns to provide the best accuracy. RFC can also tell us how much each feature contributes to class prediction using Feature Importance and tree graphs for better interpretation.

Overall saying Random Forest Classifier performs better with more categorical data than numeric and logistic regression is a little confusing when comes to categorical data So. If the dataset has more Categorical data and consists of outliers it is better to use Random Forest Classifier.

According to Google:

Logistic regression performs better when the number of noise variables is less than or equal to the number of explanatory variables and the random forest has a higher true and false positive rate as the number of explanatory variables increases in a dataset.


