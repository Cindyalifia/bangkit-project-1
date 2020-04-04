# Import library and dataset
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from IPython.display import clear_output

import tensorflow as tf # version=2.1.0


# Import dataset
data=pd.read_csv('fake_job_postings.csv')


# Selecting feature
# Drop unused field to making a model
df = data.drop(columns=['job_id','title','location','department','company_profile','description','requirements','benefits'])


# check target nan value in the data 
combined = df.copy()
nan_percentage = combined.isnull().sum().sort_values(ascending=False) / combined.shape[0]
missing_val = nan_percentage[nan_percentage > 0]

plt.figure(figsize=(5,7))
sns.barplot(x=missing_val.index.values, y=missing_val.values * 100, palette="Reds_r");
plt.title("Percentage of missing values in data");
plt.ylabel("%");
plt.xticks(rotation=90);


# Taking care NaN value 
# Drop column salary range 
df = df.drop(['salary_range'], axis=1)


# Replace NaN with '-' string
df = df.fillna('-')


# Encode categorical variable from the data
pd.DataFrame([{'employment_type': len(df['employment_type'].value_counts()),    
               'required_experience': len(df['required_experience'].value_counts()),
               'required_education': len(df['required_education'].value_counts()),
               'industry': len(df['industry'].value_counts()),
               'function': len(df['function'].value_counts()),
              }], columns = ['employment_type', 'required_experience', 'required_education', 'industry', 'function'], 
              index = ['quantity of unique value'])


# Creating feature columns from our data
CATEGORICAL_COLUMNS = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
NUMERIC_COLUMNS = ['telecommuting', 'has_company_logo']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# Splitting the data
X=df.drop("fraudulent", axis=1)
y=df["fraudulent"]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=7)


# Define input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(X_train, y_train)
eval_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

