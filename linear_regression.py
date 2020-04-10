import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#IMPORTING LIBRARIES


#%%
df=pd.read_csv("Advertising.csv")
#IMPORTING CSV FILE
df.head()
#FIRST 5 ROWS



#%%
plt.figure(figsize=(12,6))
plt.scatter(df['TV'],df['sales'])
plt.xlabel("TV ADS")
plt.ylabel("SALES")
plt.show()
#VISUALISING DATA




#%%
x=df['TV'].values.reshape(-1,1)
y=df['sales'].values.reshape(-1,1)
# X AND Y AXIS 



#%%
lr=LinearRegression()
lr.fit(x,y)
#FITTING LINEAR REGRESSION TO THE DATASET



#%%
predict=lr.predict(50)
print(predict)
#PREDICTION



#%%
predictions=lr.predict(x)
#PREDICTING Y VALUES WITH LINEAR REGRESSION

plt.figure(figsize=(12,6))
plt.scatter(df['TV'],df['sales'],c="black")
plt.plot(df['TV'],predictions,c="red",linewidth=2)
plt.xlabel("TV ADS")
plt.ylabel("SALES")
plt.title("Linear Regression")
plt.show()
#VISUALISING LINEAR REGRESSION RESULTS


#%%

b0_=lr.intercept_
b1=lr.coef_

print("Intercept: ",b0_)
print("Coef: ",b1)





















