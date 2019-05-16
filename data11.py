
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
#2. Veri Onisleme

#2.1. Veri Yukleme
datas = pd.read_csv('data1.csv')
#pd.read_csv("veriler.csv")

from sklearn.model_selection import train_test_split

#verilerin egitim ve test icin bolunmesi



x_train, x_test,y_train,y_test = train_test_split(datas.iloc[:,:-1],datas.iloc[:,-1:],test_size=0.33, random_state=0)




from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)

y_pred1 = regressor.predict(x_test)

print(y_pred1)

katilimci =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]



x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train)
y_pred2 = regressor.predict(x_test)



#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)

X = x_train.values
Y = y_train.values

X_test = x_test.values
Y_test = y_test.values


x_poly = poly_reg.fit_transform(X)
 
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
 
poly_pred = lin_reg2.predict(poly_reg.fit_transform(X_test))

 
 




plt.title("pred dec.tree(green), pred mul.lin.reg.(blue) support vector reg (yellow) vs real (red)")
plt.xlabel("katilimci")
plt.ylabel("score (tons)")

plt.plot(katilimci,y_pred1, color='green')
plt.plot(katilimci,y_pred2, color='yellow')
plt.plot(katilimci,y_pred, color='blue')
plt.plot(katilimci,y_test, color='red')

plt.plot(katilimci,poly_pred, color='pink')

plt.show()

