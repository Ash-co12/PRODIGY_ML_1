import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

#Reading Training csv dataset using pandas
try:
    #Taking necessary columns from dataset
    tds = pnd.read_csv("train.csv", usecols=["LotArea", "SalePrice","FullBath","BedroomAbvGr"])
except FileNotFoundError:
    print("File not found.")
    exit()

#Training The model
x = tds[["LotArea","FullBath","BedroomAbvGr"]]
y = tds["SalePrice"]
reg.fit(x,y)

#Creating a new panda frame and testing a new test data and saving it, for larger predictions
ts = pnd.read_csv("test.csv", usecols=["LotArea","FullBath","BedroomAbvGr"])
tl = reg.predict(ts)
ts["SalePrice"] = tl
ts.to_csv('test_result.csv',index=False)