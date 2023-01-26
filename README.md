
# Predicting Credit Risk using Supervised Machine Learning Model

## Brief Introduction -
Lending services companies allow individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. Data on previous loan applications (both approved and Rejected) will be used to build Machine Learning Models to classify the risk level of given loans. 

The archieved dataset had information on loan_size,	interest_rate,	borrower_income,	debt_to_income,	num_of_accounts,	derogatory_marks,	total_debt,	loan_status. This a binary classification ML problem with loan_status either likely to be 0 (not approved) or 1 (approved).

## Aims and Objective -
Employing Logistic Regression model and Random Forest Classifier, the loan_status of the appliaction will be predicted. 

      (1) Performance of the models will be asessed with a preprocessing step included (for Logistic Regression model & 
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
![LR](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/new_LR.png)


Linear Regression performed with a accuracy scores of 0.9921. However, with the scaling of the data it improved even further to 0.9941. If this was a model with a score of say., 0.7823, then there is a high likelihood of this score increasing further, with a preprocessing step such as MinMaxScaler or StandardScaler included.
This increase in accuracy score was also reflected in Recall and f1-scores' further improvement.

## Random Forest Classifier with or without Preprocessing -
![RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/RFC.png)

Similar to Linear Regression, Random Forest Classifier also performed really well; with an accuracy score of over 0.99.

Unlike Linear Regression which relies on the linear relationship between input features and output variables; the standardization/scaling remain an important step to improving its performance when the feature values are not scaled.

Random Forest Classifier donot rely on the linear relationship of feature and output variable values, and therefore there was little to no improvement in the performance of this model with a scaling step included. 

## Feature Selection and evaluation of Random Forest Classifier
![FS-RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Feature%20Selection%20-%20scores.png)

While developing an accurate ML model, it is essential that features that are included in the model, contribute to the overall performance of the model and its accurate prediction. 

Eventhough, these model (demonstarted above) have already performed so well with an accuracy score nearing a perfect score; further refinement of this model could potentially improve Precision, Recall, f1-scores. With this in mind, feature imortance was attempted using SelectFromModel. Among the input features, interest_rate, borrower_income, and debt_to_income were identified to contribute to the overall predictability of the model. 

Further refinement of the model was carried out using these three selected features. This didnot improve the score of a already well performing model. However, this refinement process is important when a model is faced with hundreds of features, many of which do not contribute to the overall performance and generate noise for the model.  

## Hyperparameter Tuning and comparison between the two models (code in Bonus file)
![LR v/s RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Hyperparameter.png)

Having recognized there is still room for improvement with Precision, Recall, f1-scores, hyperparameter tuning was carried out using GridSearchCV. This improved the Recall and f1-score to near perfect scores. 

If this was a model with an accuracy score much lower than 0.99, then there is a high likelihood of this score also increasing further, with hyperparameter tuning step included.



 





