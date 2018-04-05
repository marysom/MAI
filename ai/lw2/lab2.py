import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import quandl 
import math
import numpy as np

data = pd.read_csv("global_co2.csv")
data.columns=['Year',       'Total',       'Gas Fuel',    'Liquid Fuel',
              'Solid Fuel', 'Cement',      'Gas Flaring', 'Per Capita']
data.info()
print('-------------------------------------')
print(data.isnull().any())
print('-------------------------------------')
print('Исходные данные:')
print(data.head())
print('-------------------------------------')
data1 = data.dropna()
print('Данные без NA')
print(data1.head())
print(data1.shape)
print('-------------------------------------')
per_capita=data['Per Capita'].values
def search_1st_notnan(per_capita):
    i=0
    while i < len(per_capita):
        if math.isnan(per_capita[i]):
            i+=1
        else:
            break
    return i

def out_of_range(per_capita):
    k=search_1st_notnan(per_capita)
    maxper=per_capita[k]
    for i in range(k+1,len(per_capita)):
        if (per_capita[i]>maxper)and(not math.isnan(per_capita[i])):
            maxper=per_capita[i]
    return maxper+0.2

def average(per_capita):
    k=search_1st_notnan(per_capita)
    sumper=0
    count=0
    for i in range(k+1, len(per_capita)):
        if not math.isnan(per_capita[i]):
            sumper+=per_capita[i]
            count+=1
    aver=sumper/count
    return aver
            
print('1st not NA = ',search_1st_notnan(per_capita))
print('-------------------------------------')
print('average = ', average(per_capita))
print('-------------------------------------')
data2=data.fillna(average(per_capita))
print('Данные с заменой NA на среднее')
print(data2.head())
print(data2.shape)
print('-------------------------------------')
print('out of range = ',out_of_range(per_capita))
print('-------------------------------------')
print('Данные с заменой NA на значение вне диапазона')
data3=data.fillna(out_of_range(per_capita))
print(data3.head())
print(data3.shape)
print('-------------------------------------')

sns_data=sns.heatmap(data.corr(), annot=True)
useful_columns = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']
sns_pair=sns.pairplot(data[useful_columns])

'''-------------------------------------------------'''
a=len(data1['Per Capita'])
a=math.floor(a*0.7)

model1 = LinearRegression()
x1 = data1[useful_columns]
y1 = data1['Per Capita']
x1_train=x1[:a]
x1_test=x1[a:]
y1_train=y1[:a]
y1_test=y1[a:]
model1.fit(x1_train, y1_train)
pred_test = model1.predict(x1_test)
pred_train=model1.predict(x1_train)

plt.figure()
plt.scatter(data1["Total"], y1, color = 'black', s=5)
plt.plot(data1["Total"][:a], pred_train, color = 'blue')
plt.plot(data1["Total"][a:], pred_test, color = 'green')
plt.xlabel('total')
plt.ylabel('per capita (без NA)')
plt.grid()
plt.show()

plt.figure()
plt.scatter(data1["Year"], y1, color = 'black', s=5)
plt.plot(data1["Year"][:a], pred_train, color = 'blue')
plt.plot(data1["Year"][a:], pred_test, color = 'green')
plt.xlabel('year') 
plt.ylabel('per capita (без NA)')
plt.grid()
plt.show()
'''-------------------------------------------------'''
a=len(data2['Per Capita'])
a=math.floor(a*0.7)

model2 = LinearRegression()
x2 = data2[useful_columns]
y2 = data2['Per Capita']
x2_train=x2[:a]
x2_test=x2[a:]
y2_train=y2[:a]
y2_test=y2[a:]
model2.fit(x2_train, y2_train)
pred_test = model2.predict(x2_test)
pred_train=model2.predict(x2_train)

plt.figure()
plt.scatter(data2["Total"], y2, color = 'black', s=5)
plt.plot(data2["Total"][:a], pred_train, color = 'blue')
plt.plot(data2["Total"][a:], pred_test, color = 'green')
plt.xlabel('total')
plt.ylabel('per capita (с заменой NA на среднее значение)')
plt.grid()
plt.show()

plt.figure()
plt.scatter(data2["Year"], y2, color = 'black', s=5)
plt.plot(data2["Year"][:a], pred_train, color = 'blue')
plt.plot(data2["Year"][a:], pred_test, color = 'green')
plt.xlabel('year') 
plt.ylabel('per capita (с заменой NA на среднее значение)')
plt.grid()
plt.show()
'''-------------------------------------------------'''
a=len(data3['Per Capita'])
a=math.floor(a*0.7)

model3 = LinearRegression()
x3 = data3[useful_columns]
y3 = data3['Per Capita']
x3_train=x3[:a]
x3_test=x3[a:]
y3_train=y3[:a]
y3_test=y3[a:]
model3.fit(x3_train, y3_train)
pred_test = model3.predict(x3_test)
pred_train=model3.predict(x3_train)

plt.figure()
plt.scatter(data3["Total"], y3, color = 'black', s=5)
plt.plot(data3["Total"][:a], pred_train, color = 'blue')
plt.plot(data3["Total"][a:], pred_test, color = 'green')
plt.xlabel('total')
plt.ylabel('per capita (с заменой NA на значение вне диапазона)')
plt.grid()
plt.show()

plt.figure()
plt.scatter(data3["Year"], y3, color = 'black', s=5)
plt.plot(data3["Year"][:a], pred_train, color = 'blue')
plt.plot(data3["Year"][a:], pred_test, color = 'green')
plt.xlabel('year') 
plt.ylabel('per capita (с заменой NA на значение вне диапазона)')
plt.grid()
plt.show()







