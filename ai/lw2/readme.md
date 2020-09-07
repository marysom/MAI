## Отчет по лабораторной работе №2
### По курсу "Искусственный интеллект"
#### Линейная регрессия
_______________________________________
##### Выполнила студентка группы 8О-304Б [Сомова Мария](https://github.com/marysom/python/tree/master/ai/lw2)
_______________________________________
### Задание
1. Построить модель линейной регрессии для выборки Global CO2 (((14-1) mod 3)+1 = 2).
2. Проверить возможность переобучения. Визуализировать все данные и полученную на выходе линию.
3. Поэксперементировать с методами обработки ситуации с неопределенностью значения признака.

### Решение
**listing**
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import quandl 
import math
import numpy as np

data = pd.read_csv("global_co2.csv")  #считываем данные
data.columns=['Year',       'Total',       'Gas Fuel',    'Liquid Fuel',
              'Solid Fuel', 'Cement',      'Gas Flaring', 'Per Capita']
data.info()  #узнаём немного больше о наших данных
print('-------------------------------------')
print(data.isnull().any()) 
print('-------------------------------------')
print('Исходные данные:')
print(data.head())
print('-------------------------------------')
```

**output**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 260 entries, 0 to 259
Data columns (total 8 columns):
Year           260 non-null int64
Total          260 non-null int64
Gas Fuel       260 non-null int64
Liquid Fuel    260 non-null int64
Solid Fuel     260 non-null int64
Cement         260 non-null int64
Gas Flaring    260 non-null int64
Per Capita     61 non-null float64
dtypes: float64(1), int64(7)
memory usage: 16.3 KB
-------------------------------------
Year           False
Total          False
Gas Fuel       False
Liquid Fuel    False
Solid Fuel     False
Cement         False
Gas Flaring    False
Per Capita      True
dtype: bool
-------------------------------------
Исходные данные:
   Year  Total  Gas Fuel  Liquid Fuel  Solid Fuel  Cement  Gas Flaring  \
0  1751      3         0            0           3       0            0   
1  1752      3         0            0           3       0            0   
2  1753      3         0            0           3       0            0   
3  1754      3         0            0           3       0            0   
4  1755      3         0            0           3       0            0   

   Per Capita  
0         NaN  
1         NaN  
2         NaN  
3         NaN  
4         NaN  
------------------------------------- 
```
Так как выборка Global CO2 содержит множество NaN в столбце "Per Capita", рассмотрим 3 варианта работы с NaN:  
1. Удалим все строки, содержащие NaN;
2. Заменим NaN на среднее значение по столбцу "Per Capita";  
3. Заменим NaN на значение вне диапазона (возьмём максимальное значение из столбца "Per Capita" + 0.2);  
  
**listing**
```python
data1 = data.dropna()  #удаляем все строки, содержащие NaN
print('Данные без NA')
print(data1.head())
print(data1.shape)
print('-------------------------------------')
per_capita=data['Per Capita'].values
def search_1st_notnan(per_capita):  #функция, выполняющая поиск первого отличного от NaN значения в per_capita
    i=0
    while i < len(per_capita):
        if math.isnan(per_capita[i]):
            i+=1
        else:
            break
    return i

def out_of_range(per_capita):  #функция, возвращающая значение большее, чем максимальный элемент из per_capita
    k=search_1st_notnan(per_capita)
    maxper=per_capita[k]
    for i in range(k+1,len(per_capita)):
        if (per_capita[i]>maxper)and(not math.isnan(per_capita[i])):
            maxper=per_capita[i]
    return maxper+0.2

def average(per_capita):  #функция, возвращающая среднее значение per_capita
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
```

**output**
```
Данные без NA
     Year  Total  Gas Fuel  Liquid Fuel  Solid Fuel  Cement  Gas Flaring  \
199  1950   1630        97          423        1070      18           23   
200  1951   1767       115          479        1129      20           24   
201  1952   1795       124          504        1119      22           26   
202  1953   1841       131          533        1125      24           27   
203  1954   1865       138          557        1116      27           27   

     Per Capita  
199        0.64  
200        0.69  
201        0.68  
202        0.69  
203        0.69  
(61, 8)
-------------------------------------
1st not NA =  199
-------------------------------------
average =  1.0616666666666668
-------------------------------------
Данные с заменой NA на среднее
   Year  Total  Gas Fuel  Liquid Fuel  Solid Fuel  Cement  Gas Flaring  \
0  1751      3         0            0           3       0            0   
1  1752      3         0            0           3       0            0   
2  1753      3         0            0           3       0            0   
3  1754      3         0            0           3       0            0   
4  1755      3         0            0           3       0            0   

   Per Capita  
0    1.061667  
1    1.061667  
2    1.061667  
3    1.061667  
4    1.061667  
(260, 8)
-------------------------------------
out of range =  1.53
-------------------------------------
Данные с заменой NA на значение вне диапазона
   Year  Total  Gas Fuel  Liquid Fuel  Solid Fuel  Cement  Gas Flaring  \
0  1751      3         0            0           3       0            0   
1  1752      3         0            0           3       0            0   
2  1753      3         0            0           3       0            0   
3  1754      3         0            0           3       0            0   
4  1755      3         0            0           3       0            0   

   Per Capita  
0        1.53  
1        1.53  
2        1.53  
3        1.53  
4        1.53  
(260, 8)
-------------------------------------
```
Визуализируем исходные данные:  

**listing**
```python
sns_data=sns.heatmap(data.corr(), annot=True)
useful_columns = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']
sns_pair=sns.pairplot(data[useful_columns])
```
Корреляционая карта исходных данных выглядит следующим образом:  

![sns_data](https://github.com/marysom/python/blob/master/ai/lw2/corr.png)
  
Рассмотрим sns_pair:  
Можно увидеть, как связаны между собой различные признаки. На диагонали матрицы графиков расположены гистограммы распределений признака. Остальные же графики — это обычные scatter plots для соответствующих пар признаков.

![sns_pair](https://github.com/marysom/python/blob/master/ai/lw2/pairplot.png)  

Будем делить выборку в соотношении 70% тренировочных данных, 30% - тестовых.  Рассмотрим две зависимости:   
1. Per Capita от Total;
2. Per Capita от Year;  

Построим модель линейной регрессии для data1 (без NaN):  

**listing**
```python
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
```
![per_capita-total](https://github.com/marysom/python/blob/master/ai/lw2/Figure_1.png)  

![per_capita-year](https://github.com/marysom/python/blob/master/ai/lw2/Figure_2.png)

Построим модель линейной регрессии для data2 (с заменой NaN на среднее значение):  

**listing**
```python
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
```

![percap-total](https://github.com/marysom/python/blob/master/ai/lw2/Figure_3.png)  

![percap-year](https://github.com/marysom/python/blob/master/ai/lw2/Figure_4.png)  

Построим модель линейной регрессии для data3 (с заменой NaN на значение вне диапазона):  

**listing**
```python
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
```

![per-total](https://github.com/marysom/python/blob/master/ai/lw2/Figure_5.png)  

![per-year](https://github.com/marysom/python/blob/master/ai/lw2/Figure_6.png)  


### Вывод
На графиках отчетливо видно, что построенная модель регрессии переобучилась на тренировочных данных, поэтому прогноз не удался.

В лабораторной работе рассматривались три возможных способа обработки NaN в выборке:  
1. Удалить все строки, содержащие NaN;
2. Заменить NaN на среднее значение по столбцу "Per Capita";  
3. Заменить NaN на значение вне диапазона.  

Наиболее удачным, на мой взгяд, получился 1 вариант - удаление всех строк, содержащих NaN, во 2 и 3 случаях такая замена NaN слишком сильно повлияла на результат, поэтому, полагаю, что наилучший вариант  - это избежать при возможности работу с NaN.
