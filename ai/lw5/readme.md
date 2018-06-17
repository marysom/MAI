### Лабораторная работа №5   
#### Задание:   
Реализовать алгоритм выявляющий взаимосвязанные сообщения на языке Python. Подобрать или создать датасет и обучить модель. Продемонстрировать зависимость качества кластеризации от объема, качества выборки и числа кластеров. Продемонстрировать работу вашего алгоритма. Обосновать выбор данного алгоритма машинного обучения. Построить облако слов для центров кластеров(wordcloud).
   
Первое, что необходимо было сделать для выполнения данной лабораторной работы - подобрать датасет. Я нашла статистику поисковых запросов в Яндексе, содержащих слово "попугай", за последний месяц: [input_data.csv](https://github.com/marysom/python/blob/master/ai/lw5/input_data.csv). В каждой строке файла находится один поисковой запрос и количество показов этого запроса в месяц. Перед тем, как применить алгоритм, файл необходмо очистить от знаков препинания, спецсимволов, в итоге получим готовый к применению алгоритма датасет, в каждой строке которого находится  один которткий поисковой запрос.      
Для реализации алгоритма я воспользовалась библиотекой NLTK. Сначала проводится нормализация -  слова приводятся к начальной форме, для этого используется стеммер Портера, затем создается матрица весов tfidf_matrix (каждый поисковой запрос -  документ). К полученной матрице применим алгоритм кластеризации.
Кластеризация выполнена методом KMeans, поскольку по заданию лабораторной работы необходимо продемонстировать зависимость качества кластеризации от количества кластеров, а метод кластеризации KMeans позволяет задать самостоятельно количество кластеров.

Листинг:   
``` python
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import feature_extraction
from sklearn.cluster import KMeans
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

def tok(data):
    tokens = [word.lower() for sent in nltk.sent_tokenize(data) for word in nltk.word_tokenize(sent)]
    filt_tk = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filt_tk.append(token)
    return filt_tk

def tok_st(data):
    tokens = [word for sent in nltk.sent_tokenize(data) for word in nltk.word_tokenize(sent)]
    filt_tk = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filt_tk.append(token)
    stems = [stemmer.stem(t) for t in filt_tk]
    return stems

def main():
    data = pd.read_csv("input_data.csv", sep = ';')
    data = data['keyword']
    print('Исходные данные:')
    print(data.head(10))
    print('Количество строк:', len(data))
    for i in range(len(data)):
        data[i] = re.sub(r'(\<(/?[^>]+)>)', ' ', data[i])
        data[i] = re.sub('[^а-яА-Я ]', '', data[i])
    print('\nОбработанные данные (исключены знаки препинания):')    
    print(data.head(10))   

    tv_st = []
    tv_tk = []
    for i in data:
        allwords_stemmed = tok_st(i)
        tv_st.extend(allwords_stemmed)
        allwords_tokenized = tok(i)
        tv_tk.extend(allwords_tokenized)
    stopwords = nltk.corpus.stopwords.words('russian')
    stopwords.extend(['бы', 'над', 'о', 'а', 'то', 'из', 'из-за', 'с', 'за', 'для', 'что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на'])
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=2050, min_df=0.01, stop_words=stopwords, use_idf=True, tokenizer=tok_st, ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)

    print('\nКластеризация осуществляется методом Kmeans\nВведите колчество кластеров:')
    count = int(input())
    if (count <= 0) or (count >= len(data) + 1):
        print('Ошибка')
    else:
        km = KMeans(n_clusters=count).fit(tfidf_matrix)
        centers = km.cluster_centers_
        idx = km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()
        clusterkm = km.labels_.tolist()
        frame = pd.DataFrame(data, index = [clusterkm])
        feature_names = tfidf_vectorizer.get_feature_names()
        out = { 'data': data, 'cluster': clusterkm }
        frame_res = pd.DataFrame(out, columns = ['data', 'cluster'])
        frame_res.to_csv("result.txt", sep='\t', encoding='utf-8')
        print('Результаты кластеризации в файле result.txt')
        print(frame_res.head(20))

if __name__ == "__main__":
    main()
```

Протокол:   
```
Исходные данные:
0              попугай
1    волнистый попугай
2       купить попугая
3      попугай корелла
4         попугай кеша
5             попугать
6      попугай попугал
7         попугай фото
8        попугаи видео
9     сколько попугаев
Name: keyword, dtype: object
Количество строк: 2050

Обработанные данные (исключены знаки препинания):
0              попугай
1    волнистый попугай
2       купить попугая
3      попугай корелла
4         попугай кеша
5             попугать
6      попугай попугал
7         попугай фото
8        попугаи видео
9     сколько попугаев
Name: keyword, dtype: object

Кластеризация осуществляется методом Kmeans
Введите колчество кластеров:
3
Результаты кластеризации в файле result.txt
                         data  cluster
0                     попугай        2
1           волнистый попугай        0
2              купить попугая        1
3             попугай корелла        0
4                попугай кеша        0
5                    попугать        2
6             попугай попугал        2
7                попугай фото        0
8               попугаи видео        0
9            сколько попугаев        0
10          говорящий попугай        0
11  говорящий попугай говорит        0
12            попугай говорит        0
13       попугаи неразлучники        2
14                   попугаев        2
15               про попугаев        2
16                попугая ару        1
17                попугай ара        2
18               попугай жако        0
19               попугай цена        0
```
Результат кластеризации в случае разбиения на 3 кластера: [result_3.txt](https://github.com/marysom/python/blob/master/ai/lw5/result_3.txt)
```
Введите колчество кластеров:
7
Результаты кластеризации в файле result.txt
                         data  cluster
0                     попугай        0
1           волнистый попугай        4
2              купить попугая        3
3             попугай корелла        1
4                попугай кеша        6
5                    попугать        0
6             попугай попугал        0
7                попугай фото        1
8               попугаи видео        1
9            сколько попугаев        1
10          говорящий попугай        1
11  говорящий попугай говорит        1
12            попугай говорит        1
13       попугаи неразлучники        0
14                   попугаев        0
15               про попугаев        0
16                попугая ару        5
17                попугай ара        0
18               попугай жако        1
19               попугай цена        1
```
Результат кластеризации в случае разбиения на 7 кластеров: [result_7.txt](https://github.com/marysom/python/blob/master/ai/lw5/result_7.txt)
```
Введите колчество кластеров:
10
Результаты кластеризации в файле result.txt
                         data  cluster
0                     попугай        0
1           волнистый попугай        9
2              купить попугая        1
3             попугай корелла        8
4                попугай кеша        4
5                    попугать        0
6             попугай попугал        0
7                попугай фото        5
8               попугаи видео        6
9            сколько попугаев        7
10          говорящий попугай        3
11  говорящий попугай говорит        3
12            попугай говорит        3
13       попугаи неразлучники        0
14                   попугаев        0
15               про попугаев        0
16                попугая ару        2
17                попугай ара        0
18               попугай жако        3
19               попугай цена        3
```
Результат кластеризации в случае разбиения на 10 кластеров: [result_10.txt](https://github.com/marysom/python/blob/master/ai/lw5/result_10.txt)

#### Вывод:   
К сожалению, добиться хороших результатов не удалось, данные не получилось четко кластеризовать. Однако прослеживается тенденция к улучшению результата кластеризации при увеличении количества кластеров. Таким образом, при решении задачи кластеризации методом KMeans нужно сначала проанализировать датасет, чтобы примерно оценить необходимое количество кластеров.
