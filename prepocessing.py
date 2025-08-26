import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

train.info()

train1 = train[['Time_spent_Alone',	'Stage_fear',	'Social_event_attendance',	'Going_outside',	'Drained_after_socializing',	'Friends_circle_size',	'Post_frequency']].copy()

for col in ['Time_spent_Alone',	'Social_event_attendance',	'Going_outside',	'Friends_circle_size',	'Post_frequency']:
  train1[col] = train1[col].fillna(train1[col].mean())

for col1 in ['Stage_fear',	'Drained_after_socializing']:
  train1[col1] = train1[col1].apply(lambda x: 0 if x == 'No' else 1 if x == 'Yes' else None)
  train1[col1] = train1[col1].fillna(train1[col1].mode()[0])

y = train['Personality']
y0 = y.apply(lambda x: 0 if x=='Introvert' else 1)

tr_X, val_X, tr_y, val_y = train_test_split(train1, y0, test_size=0.1, random_state=42)
