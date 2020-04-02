import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#%%

df=pd.read_csv("Advertising.csv")

df.head()

#%%

plt.figure(figsize=(12,6))
plt.scatter(df['TV'],df['sales'])
plt.xlabel("Tv Reklamlarına Harcanan Para")
plt.ylabel("Satışlar")
plt.title("Tv Reklamlarına Yapılan Yatırım - Satışlar")
plt.show()

#%%

x=df['TV'].values.reshape(-1,1)
y=df['sales'].values.reshape(-1,1)

#%%

lr=LinearRegression()
lr.fit(x,y)

#%%

tahmin=lr.predict(50)
print(tahmin)

#%%
tahminler=lr.predict(x)

plt.figure(figsize=(12,6))
plt.scatter(df['TV'],df['sales'],c="black")

plt.plot(df['TV'],tahminler,c="red",linewidth=2)

plt.xlabel("Tv Reklamlarına Harcanan Para")
plt.ylabel("Satışlar")
plt.show()

#%%

b0_=lr.intercept_
b1=lr.coef_

print("Intercept: ",b0_)
print("Coef: ",b1)





















