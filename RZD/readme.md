*From R to Python*  
  
**Входные данные: [newdata.csv](https://github.com/marysom/python/blob/master/RZD/newdata.csv)**  
Обозначения:    
  path - путь от последнего ПТО до места схода  
  year - год изготовления боковой рамы  
  days - количество дней от последнего ремонта (преимущественно от ДР)  

**Код: [RZD.py](https://github.com/marysom/python/blob/master/RZD/RZD.py)**
  
**Результат:**  
Связь признаков. На диагонали матрицы графиков расположены гистограммы распределений признака. Остальные графики — это обычные scatter plots для соответствующих пар признаков.
![pair](https://github.com/marysom/python/blob/master/RZD/pairplots.jpg)  
  
Корреляционая карта исходных данных выглядит следующим образом:
![corr](https://github.com/marysom/python/blob/master/RZD/corr.png)  
  
Далее рассматривается зависимость days от path  

Плотность:  
![den](https://github.com/marysom/python/blob/master/RZD/den.jpg)  
  
![years](https://github.com/marysom/python/blob/master/RZD/years.png) 

Линейная регрессия (выборка данных разделена на тестовую/тренировочную в отношении 80%/20%):  
![regr](https://github.com/marysom/python/blob/master/RZD/regr.png)  

Кластеризация методом Kmeans (количество кластеров задается вручную):  
![kmeans](https://github.com/marysom/python/blob/master/RZD/kmeans.png)  

Карта рисков:  
![risk1](https://github.com/marysom/python/blob/master/RZD/risk1.png)
![risk2](https://github.com/marysom/python/blob/master/RZD/risk2.png)
