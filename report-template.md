# Credit Risk Classification Report

## Overview of the Analysis

1. ### Purpose Of Analysis

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Analysis, we are using various techniques to train and evaluate models with imbalanced classes.

2. ### Dataset Used and Prediction Variables

 * Here we are using a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
 * The Input features (X) of the data are :
      - loan_size
      - interest_rate
      - borrower_income	
      - debt_to_income	
      - num_of_accounts	
      - derogatory_marks
      - total_debt 
 * The Output label (y) is `loan_status` column
     * A value of 0 in the “loan_status” column means that the loan is healthy.
     * A value of 1 means that the loan has a high risk of defaulting.

3. ### Stages of Machine Learning Process

  1. Data Collection
    As mentioned in the previous section,  we used dataset of historical lending activity from a peer-to-peer lending sercices company.
  2. Data Preprocessing
    Here we are splititing the dataset as training set and testing set.
  3. Choosing a model
    According to our problem, we need to categorize the riskiness of the loans as “high risk” and “low risk”. We use `logistic regression` as our model.
  4. Training
    Using `logistic regression` model we fit the training data.
  5. Prediction
    We then use the trained model to predict the values for testing data.
  6. Evaluation
    We then evaluate the model's performance by doing the following:
      * Calculate the accuracy score of the model.
      * Generate a confusion matrix.
      * Generate classification report.
   7. Resampling Training Data
     We Oversample the training data because out data is Imbalanced Data(The dataset has fewer sample points for high risk loans.) Here we use `RandomOverSampler` as our resampling technique.
   8. Train, Test and Evaluate the model with Resampled data
     Our `logistic regression`  model is then trained with resampled data and is then used to predict the values for testing data. We then generate evaluation metrics for this model.
     
4. ### Methods Used

* ####`train_test_split`
  Used to split the given dataset into training and testing data.
    
* ####`LogisticRegression`
  `LogisticRegression`  model mathematically determines the probability of the sample belonging to a class. If the probability is greater than a certain cutoff point, the model assigns the sample to that class. If the probability is less than or equal to the cutoff point, the model assigns the sample to the other class.
    
* ####`balanced_accuracy_score`
  It is an evaluation metric in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
   
* ####`confusion_matrix`
  Compute confusion matrix to evaluate the accuracy of a classification

* ####`classification_report_imbalanced`
  Build a classification report based on metrics used with imbalanced dataset. This report compiles the state-of-the-art metrics: precision/recall/specificity, geometric mean, and index balanced accuracy of the geometric mean.

* ####`RandomOverSampler`
  To over-sample the minority class(es) by picking samples at random with replacement.


---
## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Linear Regression Model with Original training data
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  * Accuracy
  * Precision
  * Recall



* Machine Learning Model 2: Linear Regression Model with Oversampled training data
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  * Accuracy
  * Precision
  * Recall

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.