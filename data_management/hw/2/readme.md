### Домашнее задание №2

#### 1. Простые выборки

- 1.1 SELECT , LIMIT - выбрать 10 записей из таблицы ratings (Для всех дальнейших запросов выбирать по 10 записей, если не указано иное)
- 1.2 WHERE, LIKE - выбрать из таблицы links всё записи, у которых imdbid оканчивается на "42", а поле movieid между 100 и 1000

#### 2. Сложные выборки: JOIN

- 2.1 INNER JOIN выбрать из таблицы links все imdbId, которым ставили рейтинг 5

#### 3. Аггрегация данных: базовые статистики

- 3.1 COUNT() Посчитать число фильмов без оценок
- 3.2 GROUP BY, HAVING вывести top-10 пользователей, у который средний рейтинг выше 3.5

#### 4. Иерархические запросы

- 4.1 Подзапросы: достать любые 10 imbdId из links у которых средний рейтинг больше 3.5.
- 4.2 Common Table Expressions: посчитать средний рейтинг по пользователям, у которых более 10 оценок.  Нужно подсчитать средний рейтинг по все пользователям, которые попали под условие - то есть в ответе должно быть одно число.

Листинг [hw2.sql](https://github.com/marysom/MAI/blob/master/data_management/hw/2/hw2.sql)
```sql
SELECT 'FIO: Somova Maria Sergeevna';

-- 1.1 запрос
SELECT *
FROM ratings
LIMIT 10
;

-- 1.2 запрос
SELECT *
FROM links
WHERE 1=1
	AND imdbid LIKE '%42'
	AND movieid between 100 and 1000
LIMIT 10
;

-- 2 запрос
SELECT 
	l.imdbid
FROM links AS l
	INNER JOIN 
	ratings AS r
		ON 
			l.movieid = r.movieid
			AND 
			r.rating = 5
LIMIT 10
;

-- 3.1 запрос
SELECT 
	COUNT(l.movieid)
FROM links AS l
	LEFT JOIN 
	ratings AS r
		ON 
			l.movieid = r.movieid 
WHERE r.rating IS NUlL
;

-- 3.2 запрос
SELECT 
	  userid
	, AVG(rating)
FROM ratings
GROUP BY userid
HAVING AVG(rating) >=3.5
ORDER BY AVG(rating) DESC
LIMIT 10
;

-- 4.1 запрос
SELECT 
	imdbid
FROM links
WHERE movieid IN(
		SELECT 
			movieid 
		FROM ratings
		GROUP BY movieid 
		HAVING AVG(rating) > 3.5)
LIMIT 10            
;

-- 4.2 запрос
WITH useful_users AS (
	SELECT 
		userid       
	FROM ratings
	GROUP BY userid
	HAVING COUNT(userid) > 10 
)
SELECT 
	AVG(r.rating) 
FROM useful_users AS u 
	INNER JOIN 
	ratings AS r
		ON 
			u.userid = r.userid
; 

```

![1](https://github.com/marysom/MAI/blob/master/data_management/hw/2/hw2_1.png)

![2](https://github.com/marysom/MAI/blob/master/data_management/hw/2/hw2_2.png)
