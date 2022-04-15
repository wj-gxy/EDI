import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import numpy as np
import heapq
nltk.download('vader_lexicon')


# ============================== Sentiment Scores ==============================


sentiment_analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text):
    scores = sentiment_analyzer.polarity_scores(text)
    return scores['compound']



def extract_social_emotion_value_1(comments):
    a = []
    for i in range(len(comments)):
        c = {}
        b = sentiment_score(comments[i])
        c["comments"] = comments[i]
        c["value"] = b
        a.append(c)
    max_emo = heapq.nlargest(5, a, lambda x: x["value"])
    min_emo = heapq.nsmallest(5, a, lambda x: x["value"])
    max_emo.extend(min_emo)
    comments = []
    for i in range(len(max_emo)):
        comments.append(max_emo[i]["comments"])

    return comments


import pandas as pd
import jieba
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
stopwords = pd.read_csv("D:/minianaconda/envs/gxy/dual emotion selection//resources/Chinese/stopword.txt",
                        index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
stopwords = stopwords.stopword.values.tolist()#转为list形式
stop_w = " ".join(stopwords)
dataset = 'Twitter'
data_dir = os.path.join('D:/minianaconda/envs/gxy/dual emotion selection/dataset', dataset)
print(data_dir)
split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r', encoding='utf-8')) for t in ['train']]#打开三个.json的文件

split_datasets = dict(zip(['test'], split_datasets))
for t, pieces in split_datasets.items():
    print(len(pieces[9]["comments_words"]))
    print(pieces[9]["content"])
    comments = extract_social_emotion_value_1(pieces[9]["comments"])
    #comments = pieces[5]["comments"][:9]
    #print(type(pieces[87]["comments_words"]))
    print(pieces[9]["label"])
    print(len(comments))
    print(comments)
    all = []
    #for i in range(len(comments)):
    #    all = all+comments[i]
    str1 = " ".join(comments)
    #print(str1)
cut_test = " ".join(jieba.cut(str1))
wordcloud = WordCloud(
    font_path="C:/Windows/Fonts/simsun.ttc",
    background_color="white",width = 1000,height=800,stopwords=stop_w).generate(str1)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
