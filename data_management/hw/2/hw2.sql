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
