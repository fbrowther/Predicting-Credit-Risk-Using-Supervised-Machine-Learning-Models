
# Predicting Credit Risk using Supervised Machine Learning Model

## Brief Introduction -
Lending services companies allow individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. Data on previous loan applications (both approved and Rejected) will be used to build Machine Learning Models to classify the risk level of given loans. 

The archieved dataset had information on loan_size,	interest_rate,	borrower_income,	debt_to_income,	num_of_accounts,	derogatory_marks,	total_debt,	loan_status. This a binary classification ML problem with loan_status either likely to be 0 (not approved) or 1 (approved).

## Aims and Objective -
Employing Logistic Regression model and Random Forest Classifier, the loan_status of the appliaction will be predicted. 

      (1) Performance of the models will be compared with a preprocessing step included for (both Logistic Regression model & 
      Random Forest Classifier).
      
      (2) Main performance metrics of ML models such as Recall, Precision, Accuracy, and f1-score will be determined and compared.
  
      (3) Any fine tuning in the performace of the models will be performed with Feature Selection and Hyperparameter tuning step.
  
## Specific Libraries and modules employed - 
  (1) Scikit-Learn - 
  
      (a) Linear model - LogisticRegression, 
      
      (b) Ensemble - RandomForestClassifier,
      
      (c) model_selection - train_test_split & GridSearchCV,
      
      (d) Preprocessing - StandardScaler,
      
      (e) Metrics - classification_report, precision_score, recall_score,
      
      (f) Feature selction - SelectFromModel

## Project Performance -
## Linear Regression with or without Preprocessing -
![LR]()


## Random Forest Classifier with or without Preprocessing -
![RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/RFC.png)


## Feature Selection and evaluation of Random Forest Classifier
![FS-RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Feature%20Selection%20-%20scores.png)

## Hyperparameter Tuning and comparison between the two models.
![LR v/s RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Hyperparameter.png)





 





