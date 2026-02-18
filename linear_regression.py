#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[6]:


df=pd.read_csv("USA_Housing.csv")
df1=df.copy()
df1["Price"] = pd.to_numeric(df1["Price"], errors="coerce")

x = df1[[
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population"
]]

y = df1["Price"]  


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

predict_price = model.predict(x_test)

Me = np.mean(y_test - predict_price)
Mse = np.mean((y_test - predict_price)**2)
Rmse = np.sqrt(Mse)

print("me: ", Me)
print("mse: ", Mse)
print("rmse: ", Rmse)


# In[ ]:




