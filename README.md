# Bangkit Group Project #1

## Make a job-posting classifier using Tensorflow Linear Classifier
It will classify the job posting are fake or not

We use ```Kaggle``` job posting dataset

## Our Group :
1. Cindy Alifia P. (https://www.github.com/Cindyalifia/)
2. Pratama Yoga S. (https://www.github.com/evanezcent/)

**Our dataset fis available on ```kaggle``` fake_job_postings.csv**

In this dataset we have 17880 rows data that contains label for each rows. The label is a binary 0 and 1 is available in column fraudulent. 0 is a label for real job posting and 1 is for  fake job posting therefore this problem  is belong to the supervised learning. 

### We do prediction of real or fake job posting using Linear Classification from tensorflow.

Because this problem is a supervised learning problem means that we already have label for each data and we want to build prediction about new data whether its fake or real job posting. We got an validation accuracy 95.5% which is very good.

### Tools that needed to be installed are :
- Tensorflow version 2.1.0
- Pandas version 0.23.4
- Numpy version 1.16.1
- Matplotlib version 2.2.3
- seaborn version 0.7.1

### Split dataset 
In machine learning, we have to split our data into data training and data validation. We must have data validation to know whether is a good model to do prediction or not. If we do prediction to our data validation and already have a score and then we think that the score of the accuracy is not good enough, so we can improve the model by tuning the hyperparameter to get the better result. In this problem we split our data to 67% data training and the rest for validation.


### We use feature column as a bridge between input and our model
- A numeric column is the simplest type of column. It is used to represent real valued features. We apply this feature to these columns ('telecommuting', 'has_company_logo', 'has_questions')

- We cannot input strings directly to a model. Instead, we must first map strings to numeric or categorical values. We apply this feature to these columns ('employment_type', 'required_experience', 'required_education', 'industry', 'function')


### Result
We got an accuracy 95.5% for the data validation, and we save our model in folder 'model'.






