import numpy as np
import pandas as pd

df=pd.read_csv('insurence.csv')

print(df.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df.drop('charges',axis=1),df['charges'],test_size=0.2,random_state=42)
print(x_train.shape)

