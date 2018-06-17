### Лабораторная работа №5   
#### Задание:   
Реализовать алгоритм выявляющий взаимосвязанные сообщения на языке Python. Подобрать или создать датасет и обучить модель. Продемонстрировать зависимость качества кластеризации от объема, качества выборки и числа кластеров. Продемонстрировать работу вашего алгоритма. Обосновать выбор данного алгоритма машинного обучения. Построить облако слов для центров кластеров(wordcloud).
   
Первое, что необходимо было сделать для выполнения данной лабораторной работы - подобрать датасет. Я нашла статистику поисковых запросов в Яндексе, содержащих слово "попугай", за последний месяц: [input_data.csv](https://github.com/marysom/python/blob/master/ai/lw5/input_data.csv).     



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
