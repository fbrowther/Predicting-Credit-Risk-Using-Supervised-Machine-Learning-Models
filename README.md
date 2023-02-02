
![CreditRisk](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/creditrisk.jpeg)
# Predicting Credit Risk Employing Supervised Machine Learning Models

## Brief Introduction -
Lending services companies allow individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. Data on previous loan applications that were either Approved and Rejected were available. This data was used to develop Machine Learning Models that will accurately predict the credit_risk of future loans. 

The dataset contained information on loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts,	derogatory_marks, total_debt, and loan_status. This a binary classification ML problem with loan_status either likely to be 0 (not approved) or 1 (approved).

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

## Model Performance -
## Linear Regression with or without Preprocessing -
![LR](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/new_LR.png)


Linear Regression model performed well with an accuracy scores of 0.9921. However, with the scaling of the data it improved even further to 0.9941. Inclusion of the preprocessing step also improved Recall and f1-scores' scores of this model.

If this was a model with a score of say., 0.7823, then there is a high likelihood of this score increasing further, with a preprocessing step included (such as MinMaxScaler or StandardScaler).

The reasons for this improvement are mainly due to - Linear Regression relying on the linear relationship between input features and output variables; any feature variables whose values are not scaled might not be able to fit well in the model and thus penalizing the model performance.

## Random Forest Classifier with or without Preprocessing -
![RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/RFC.png)

Similar to Linear Regression, Random Forest Classifier also performed really well; with an accuracy score of over 0.99.

Unlike Linear Regression, Random Forest Classifier donot rely on the linear relationship of feature and output variable values, and therefore there was little to no improvement in the performance of this model with a scaling step included.

## Feature Selection and evaluation of Random Forest Classifier
![FS-RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Feature%20Selection%20-%20scores.png)

While developing a ML model, it is essential that features that are included in the model, contribute to the overall performance of the model and its accurate prediction. 

Eventhough, these model (demonstarted above) have already performed well with an accuracy score nearly perfect; any further refinement of this model could potentially improve Precision, Recall, f1-scores. With this in mind, feature imortance was attempted using SelectFromModel. Among the input features, interest_rate, borrower_income, and debt_to_income were identified to contribute the most to the overall predictability of the model.

Further refinement of the model was carried out using these three selected features. This didnot improve the score of an already well performing model. However, this refinement process is of utmost importance when a model is faced with hundreds of features, many of which do not contribute to the overall performance but might merely generate noise for the model.  

## Hyperparameter Tuning and comparison between the two models (code in Bonus file)
![LR v/s RFC](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Hyperparameter.png)

Having recognized that there is still room for improvement with Precision, Recall, f1-scores, hyperparameter tuning was carried out using GridSearchCV. This improved the Recall and f1-score to near perfect scores. 

If this was a model with an accuracy score much lower than 0.99, then there is a high likelihood of this tuning step improving the accuracy score further.

## Summary and conclusions-

        (1) Linear regression model with Preprocessing steps (StandardScaler, and GridsearchCV) and Random Forest Classifier 
            with Feature Selection were the top performing models that were developed, as a part of this Machine Learning Project.

        (2) These two models had near perfect score for Recall, f1-score and Accuracy.

        (3) These models were able to predict credit risk of current loans with 0.99 accuracy.
            

## Limitations and Discussion -

       (1) The dataset used for this modeling was imbalanced (the ratio being 1:30; 2500 for Approved to 75036 Rejected). 
            Hence, the distribution of output classes was not equal. 
    
       (2) When working with an imbalanced classification problem such as this one, the minority class (1=not approved) is typically of most importance. 

       (3) In order to rebalance the class districbution in the dataset, random resampling technique can be used. 
            The models are already performing well and therefore random resampling was not attempted in this case. 
            However, closer monitoring of the model performance on newer data should be used as a means to assess if random resampling is necessary 
            in future.
            

## Which Model to use for future predictions? -

      (1) For deployment purposes, it is recommended that Linear Regression (with Preprocesing Step and/or Hyperparameter Tuning incorportated) be chosen.
      
      (2) This model generated the highest possible scores for Recall, Accuracy, f1 Score and Precision just after Standard Scaling. 
          This score was as good as that obtained after hyperparameter tuning. 
          
      (3) Kirasich et al., found that when the variance in the explanatory and noise variables increases in a dataset, logistic regression consistently
          performs better with a higher overall accuracy compared to random forest (Kirasich, 2018). 
          
      (4) The deployable ML model needs to be robust enough in handling any variations (in both explanatory and noise variables) in the new incoming data
          and hence Logistic Regression will prove more robust in handling such changes, if they were to occur. 
          
      (5) The dataset employed in this project was 100% numerical and hence Logistic Regression will do better in this case overall as long as new 
          incoming data are checked for outliers before feeding into the model. 
          
      (6) Random Forest classifier is also a great model that can perform relatively well with dataset containing both categorical and numerical data. 
          RFC can also do very well with datasets with complex data structures resulting in complex desision boundary.
          With RFCs there is a problem of high false positive rate (which can result in low Precision!) in imbalanced dataset similar to this one. 
          Eventhough RFC performed really well in this datset, it will be reserved for a more suitable use case scenario and will not be deployed, as a  
          part of this project. 

![Credit](https://github.com/fbrowther/Supervised_ML_Models-Predicting_Credit_Risk/blob/main/Screenshots/Credit.jpeg)

## References

(1) Kirasich, Kaitlin; Smith, Trace; and Sadler, Bivin (2018) "Random Forest vs Logistic Regression: Binary Classification for Heterogeneous Datasets," SMU Data Science Review: Vol. 1: No. 3, Article 9.

Available at: https://scholar.smu.edu/datasciencereview/vol1/iss3/9

