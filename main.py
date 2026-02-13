import numpy as np
import pandas as pd

df=pd.read_csv('insurence.csv')

print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

x_train,x_test,y_train,y_test=train_test_split(df.drop('charges',axis=1),df['charges'],test_size=0.2,random_state=42)
print(x_train.shape)

print(df.isnull().sum())

oe = OrdinalEncoder(categories=[['female', 'male']])
x_train_sex = oe.fit_transform(x_train[['sex']])
x_test_sex = oe.transform(x_test[['sex']])

print(x_train_sex.shape)

ohe = OneHotEncoder(drop='first', sparse_output=False)
x_train_smoker_region = ohe.fit_transform(x_train[['smoker', 'region']])
x_test_smoker_region = ohe.transform(x_test[['smoker', 'region']])

print(x_train_smoker_region.shape)

x_train_age = x_train.drop(columns =
['sex' , 'smoker' , 'region' ]). values
x_test_age = x_test.drop(columns =
['sex' , 'smoker' , 'region']).values

print(x_train_age.shape)
print(x_test_age.shape)

x_train_transformed = np.concatenate((x_train_age , x_train_sex , x_train_smoker_region) , axis = 1)
x_test_transformed = np.concatenate((x_test_age , x_test_sex , x_test_smoker_region) , axis = 1)

print(x_train_transformed.shape)
print(x_test_transformed.shape)

from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[
('tnf2',OrdinalEncoder(categories=[['female', 'male']]), ['sex']),
('tnf3',OneHotEncoder(sparse_output=False, drop='first'), ['smoker','region'])
],remainder='passthrough')

print(transformer.fit_transform(x_train).shape)
print(transformer.transform(x_test).shape)
