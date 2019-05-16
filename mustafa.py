
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
#2. Veri Onisleme

#2.1. Veri Yukleme
datas = pd.read_csv("data1.csv)
#pd.read_csv("veriler.csv")

from sklearn.model_selection import train_test_split

#verilerin egitim ve test icin bolunmesi



x_train, x_test,y_train,y_test = train_test_split(datas.iloc[:,:-1],datas.iloc[:,-1:],test_size=0.33, random_state=0)


