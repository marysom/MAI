from numpy import *
from datetime import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines

data = pd.read_csv("newdata.csv", ';')
data.columns=['path','year','days']
print(data.head())
print(data.info())
print(data.isnull().any()) 

path=data['path'].values

data1=data.dropna()
print('Данные без NA (отсортированные)')
data1=data1.sort_values('path', ascending=True)
print(data1.head())


def search_1st_notnan(path):  
    i=0
    while i < len(path):
        if math.isnan(path[i]):
            i+=1
        else:
            break
    return i

def out_of_range(path):  
    k=search_1st_notnan(path)
    maxp=path[k]
    for i in range(k+1,len(path)):
        if (path[i]>maxp)and(not math.isnan(path[i])):
            maxp=path[i]
    return maxp+1.

def average(path):  
    k=search_1st_notnan(path)
    sump=0
    count=0
    for i in range(k+1, len(path)):
        if not math.isnan(path[i]):
            sump+=path[i]
            count+=1
    aver=sump/count
    return aver

def search_max(days,a):
    maxd=0
    for i in range(a):
        if days[i]>maxd:
            maxd=days[i]
    return maxd



sns_data=sns.heatmap(data1.corr(), annot=True)
sns_pair=sns.pairplot(data1)

sns.set(style="dark")
f, axes = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
x, y = data1['path'],data1['days']
sns.kdeplot(x, y, cmap=cmap, shade=True)
f.tight_layout()
plt.xlim(0,1500)
plt.ylim(0,1500)
plt.show()

a=len(data1['path'])
a_yel=math.floor(a/7)
a_or=math.floor(a/3)
a_red=math.floor(a)
path1=data1['path'].values
days1=data1['days'].values

matplotlib.rcParams["lines.linewidth"] = 1
matplotlib.rcParams["lines.linestyle"] = "--"

plt.figure()
plt.scatter(data1['path'], data1['days'], color='black', s=5)

plt.plot([0., path1[0]-5.],[days1[0]-5.,days1[0]-5.], color='green')
plt.plot([path1[0]-5.,path1[0]-5.],[0.,days1[0]-5.],color='green')

plt.plot([0.,path1[a_yel]],[search_max(days1,a_yel),search_max(days1,a_yel)],color='yellow')
plt.plot([path1[a_yel],path1[a_yel]],[0.,search_max(days1,a_yel)],color='yellow')

plt.plot([0.,path1[a_or]],[search_max(days1,a_or),search_max(days1,a_or)],color='orange')
plt.plot([path1[a_or],path1[a_or]],[0.,search_max(days1,a_or)],color='orange')

plt.plot([0.,max(path1)+5.],[max(days1)+5.,max(days1)+5.],color='red')
plt.plot([max(path1)+5.,max(path1)+5.],[0.,max(days1)+5.],color='red')

plt.xlim(0.,max(path1)+100)
plt.ylim(0.,max(days1)+100)
plt.show()

plt.figure()
for i in range(a_yel+1):
    plt.scatter(path1[i],days1[i], color='yellow', s=5)
for i in range(a_yel+1,a_or):
    plt.scatter(path1[i],days1[i], color='orange', s=5)
for i in range(a_or,a_red):
    plt.scatter(path1[i],days1[i], color='red', s=5)

plt.plot([0., path1[0]-5.],[days1[0]-5.,days1[0]-5.], color='green')
plt.plot([path1[0]-5.,path1[0]-5.],[0.,days1[0]-5.],color='green')

plt.plot([0.,path1[a_yel]],[search_max(days1,a_yel),search_max(days1,a_yel)],color='yellow')
plt.plot([path1[a_yel],path1[a_yel]],[0.,search_max(days1,a_yel)],color='yellow')

plt.plot([0.,path1[a_or]],[search_max(days1,a_or)+5.,search_max(days1,a_or)+5.],color='orange')
plt.plot([path1[a_or],path1[a_or]],[0.,search_max(days1,a_or)+5.],color='orange')

plt.plot([0.,max(path1)+5.],[max(days1)+7.,max(days1)+7.],color='red')
plt.plot([max(path1)+5.,max(path1)+5.],[0.,max(days1)+7.],color='red')

plt.xlim(0.,max(path1)+100)
plt.ylim(0.,max(days1)+100)
plt.show()


useful_col=['path','year']
x1 = data1[useful_col]

#РЕГРЕССИЯ
from sklearn import linear_model
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('path','days', data=data1)
plt.ylabel('Response')
plt.xlabel('Explanatory')
linear = linear_model.LinearRegression()
trainX = np.asarray(path1[45:]).reshape(-1, 1)
trainY = np.asarray(days1[45:]).reshape(-1, 1)
testX = np.asarray(path1[:45]).reshape(-1, 1)
testY = np.asarray(days1[:45]).reshape(-1, 1)
linear.fit(trainX, trainY)
linear.score(trainX, trainY)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('R² Value: \n', linear.score(trainX, trainY))
predicted = linear.predict(testX)



sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('path','days', scatter=True, fit_reg=False, data=data1, hue='year')
plt.ylabel('days')
plt.xlabel('path')



#КЛАСТЕРИЗАЦИЯ
from sklearn.cluster import KMeans
#from sklearn.cross_validation import train_test_split
kmeans = KMeans(n_clusters=4)
X = x1
kmeans.fit(X)
data1['Pred'] = kmeans.predict(X)
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.lmplot('path','days', scatter=True, fit_reg=False, data=data1, hue = 'Pred')
plt.show()




