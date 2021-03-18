# Hr_Analytics
To predict the probability of a candidate to look for a new job or will stay with the current employer.


Problem Understanding
A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some of the courses provided by the company. Many people signup for their training. Company wants to analyze which of these candidates would want to work for the company after training, or will they look for a new employment opportunity somewhere else. these analysis are important for the comapany because it helps to reduce the cost and time as well as improve the quality of training or planning the courses better. Information related to demographics, education, experience are in hands from candidates signup and enrollment.

Data sources
For the purpose analysis, Data used in this project was taken from kaggle. below is the link to the dataset.
https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

Data Cleaning
There were several problems with this data that needed to be fixed. • The dataset has so many missing values in some of the attributes. I treated the missing data using mode function as much of the variables were categorical. after performing chi_square and independed sample t-test "major_discipline", "enrolled_id" & "gender" were removed. "city" was dropped beacuse it had too many values which could have made our model a little expensive when encoded.

Modeling
The problem at hand was of classification, hence I chose two famous classification algorithm’s, Logistic Regression and Decision tree to make a ML model.

Evaluation
Logistic regression predicted with an accuracy of 76% and Decision Tree wit an 
overall accuracy of 80%, although AUC of both the models remained the same.
Hence, we will go with the Decision Tree model
