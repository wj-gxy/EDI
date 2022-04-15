import joblib
import pandas as pd
import numpy as np
import jieba
import heapq
import string

# ============================== Category ==============================
baidu_emotions = ['angry', 'disgusting', 'fearful',
                  'happy', 'sad', 'neutral', 'pessimistic', 'optimistic']
baidu_emotions.sort()

baidu_emotions_2_index = dict(
    zip(baidu_emotions, [i for i in range(len(baidu_emotions))]))


def baidu_arr(emotions_dict):
    arr = np.zeros(len(baidu_emotions))

    if emotions_dict is None:
        return arr

    for k, v in emotions_dict.items():
        # like -> happy
        if k == 'like':
            arr[baidu_emotions_2_index['happy']] += v
        else:
            arr[baidu_emotions_2_index[k]] += v

    return arr


# ============================== Lexicon and Intensity ==============================


# load negation words
negation_words = []
with open('D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/others/negative/negationWords.txt', 'r',
          encoding='utf-8') as src:
    lines = src.readlines()
    for line in lines:
        negation_words.append(line.strip())

print('\nThe num of negation words: ', len(negation_words))

# load degree words
how_words_dict = dict()
with open('D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/HowNet/intensifierWords.txt', 'r',
          encoding='utf-8') as src:
    lines = src.readlines()
    for line in lines:
        how_word = line.strip().split()
        how_words_dict[' '.join(how_word[:-1])] = float(how_word[-1])

print('The num of degree words: ', len(how_words_dict),
      '. eg: ', list(how_words_dict.items())[0])


# negation value and degree value
def get_not_and_how_value(cut_words, i, windows):
    not_cnt = 0
    how_v = 1

    left = 0 if (i - windows) < 0 else (i - windows)
    window_text = ' '.join(cut_words[left:i])

    for w in negation_words:
        if w in window_text:
            not_cnt += 1
    for w in how_words_dict.keys():
        if w in window_text:
            how_v *= how_words_dict[w]

    # for w in cut_words[left:i]:
    #     if w in negation_words:
    #         not_cnt += 1
    #     if w in how_words_dict:
    #         how_v *= how_words_dict[w]

    return (-1) ** not_cnt, how_v


_, words2array = joblib.load(
    'D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/大连理工大学情感词汇本体库/preprocess/words2array_27351.pkl')
print('[Dalianligong]\tThere are {} words, the dimension is {}'.format(
    len(words2array), words2array['快乐'].shape))


def dalianligong_arr(cut_words, windows=2):
    arr = np.zeros(29)

    for i, word in enumerate(cut_words):
        if word in words2array:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            arr += not_v * how_v * words2array[word]

    return arr


# ============================== Sentiment Scores ==============================


boson_words_dict = dict()
with open('D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/BosonNLP/BosonNLP_sentiment_score.txt',
          'r', encoding='utf-8') as src:
    lines = src.readlines()
    for line in lines:
        boson_word = line.strip().split()
        if len(boson_word) != 2:
            # print(line)
            continue
        else:
            boson_words_dict[boson_word[0]] = float(boson_word[1])
print('[BosonNLP]\t There are {} words'.format(len(boson_words_dict)))


def boson_value(cut_words, windows=2):
    value = 0

    for i, word in enumerate(cut_words):
        if word in boson_words_dict:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            value += not_v * how_v * boson_words_dict[word]

    return value


# ============================== Auxilary Features ==============================


# Emoticon
emoticon_df = pd.read_csv(
    'D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/others/emoticon/emoticon.csv')
emoticons = emoticon_df['emoticon'].tolist()
emoticon_types = list(set(emoticon_df['label'].tolist()))
emoticon_types.sort()
emoticon2label = dict(
    zip(emoticon_df['emoticon'].tolist(), emoticon_df['label'].tolist()))
emoticon2index = dict(
    zip(emoticon_types, [i for i in range(len(emoticon_types))]))

print('[Emoticon]\tThere are {} emoticons, including {} categories'.format(
    len(emoticons), len(emoticon_types)))


def emoticon_arr(text, cut_words):
    arr = np.zeros(len(emoticon_types))

    if len(cut_words) == 0:
        return arr

    for i, emoticon in enumerate(emoticons):
        if emoticon in text:
            arr[emoticon2index[emoticon2label[emoticon]]
            ] += text.count(emoticon)

    return arr / len(cut_words)


# Punctuation
def symbols_count(text):
    excl = (text.count('!') + text.count('！')) / len(text)
    ques = (text.count('?') + text.count('？')) / len(text)
    comma = (text.count(',') + text.count('，')) / len(text)
    dot = (text.count('.') + text.count('。')) / len(text)
    ellip = (text.count('..') + text.count('。。')) / len(text)

    return excl, ques, comma, dot, ellip


# Sentimental Words
def init_words(file):
    with open(file, 'r', encoding='utf-8') as src:
        words = src.readlines()
        words = [l.strip() for l in words]
    # print('File: {}, Words_sz = {}'.format(file.split('/')[-1], len(words)))
    return list(set(words))


pos_words = init_words('D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/HowNet/正面情感词语（中文）.txt')
pos_words += init_words('D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/HowNet/正面评价词语（中文）.txt')
neg_words = init_words('D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/HowNet/负面情感词语（中文）.txt')
neg_words += init_words('D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/HowNet/负面评价词语（中文）.txt')

pos_words = set(pos_words)
neg_words = set(neg_words)
print('[HowNet]\tThere are {} positive words and {} negative words'.format(
    len(pos_words), len(neg_words)))


def sentiment_words_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0, 0]

    # positive and negative words
    sentiment = []
    for words in [pos_words, neg_words]:
        c = 0
        for word in words:
            if word in cut_words:
                # print(word)
                c += 1
        sentiment.append(c)
    sentiment = [c / len(cut_words) for c in sentiment]

    # degree words
    degree = 0
    for word in how_words_dict:
        if word in cut_words:
            # print(word)
            degree += how_words_dict[word]

    # negation words
    negation = 0
    for word in negation_words:
        negation += cut_words.count(word)
    negation /= len(cut_words)

    sentiment += [degree, negation]

    return sentiment


# Personal Pronoun
first_pronoun = init_words(
    'D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/others/pronoun/1-personal-pronoun.txt')
second_pronoun = init_words(
    'D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/others/pronoun/2-personal-pronoun.txt')
third_pronoun = init_words(
    'D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/others/pronoun/3-personal-pronoun.txt')
pronoun_words = [first_pronoun, second_pronoun, third_pronoun]


def pronoun_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0]

    pronoun = []
    for words in pronoun_words:
        c = 0
        for word in words:
            c += cut_words.count(word)
        pronoun.append(c)

    return [c / len(cut_words) for c in pronoun]


# Auxilary Features辅助特征
def auxilary_features(text, cut_words):
    arr = np.zeros(17)

    arr[:5] = emoticon_arr(text, cut_words)
    arr[5:10] = symbols_count(text)
    arr[10:14] = sentiment_words_count(cut_words)
    arr[14:17] = pronoun_count(cut_words)

    return arr


# ============================== Main ==============================
import re
from zhon.hanzi import punctuation


def cut_words_from_text(text):  # 分词
    # text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    punctuation_string = string.punctuation
    for i in punctuation_string:
        text = text.replace(i, '')
    for i in punctuation:
        text = text.replace(i, '')
    return list(jieba.cut(text))


stopwords = pd.read_csv("D:/minianaconda/envs/gxy/dual emotion selection/resources/Chinese/stopword.txt",
                        index_col=False, sep="\t", quoting=3,
                        names=['stopword'], encoding='utf-8')
stopwords = stopwords.stopword.values.tolist()  # 转为list形式
test1 = []


def extract_social_emotion_value(comments_words, comments):
    a = []
    comment = []

    for i in range(len(comments_words)):
        c = {}
        b = boson_value(comments_words[i])
        c["comments_words"] = comments_words[i]
        c["comments"] = comments[i]
        c["value"] = b
        a.append(c)
    max_emo = heapq.nlargest(50, a, lambda x: x["value"])
    min_emo = heapq.nsmallest(50, a, lambda x: x["value"])
    max_emo.extend(min_emo)
    comments_word = []
    for i in range(len(max_emo)):
        comments_word.append(max_emo[i]["comments_words"])
        comment.append((max_emo[i]["comments"]))
    return comment, comments_word


def extract_social_emotion_value_1(comments_words):
    a = []
    for i in range(len(comments_words)):
        c = {}
        b = boson_value(comments_words[i])
        c["comments_words"] = comments_words[i]
        c["value"] = b
        a.append(c)
    max_emo = heapq.nlargest(50, a, lambda x: x["value"])
    min_emo = heapq.nsmallest(50, a, lambda x: x["value"])
    max_emo.extend(min_emo)
    comments = []
    for i in range(len(max_emo)):
        comments.append(max_emo[i]["comments_words"])

    return comments


'''
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
stopwords = pd.read_csv("D:/minianaconda/envs/gxy/dual emotion selection//resources/Chinese/stopword.txt",
                        index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
stopwords = stopwords.stopword.values.tolist()#转为list形式
stop_w = " ".join(stopwords)
dataset = 'Weibo-20'
data_dir = os.path.join('D:/minianaconda/envs/gxy/dual emotion selection/dataset', dataset)
print(data_dir)
split_datasets = [json.load(open(os.path.join(data_dir, '{}.json'.format(t)), 'r', encoding='utf-8')) for t in ['test']]#打开三个.json的文件
split_datasets = dict(zip(['test'], split_datasets))
for t, pieces in split_datasets.items():
    print(pieces[4]["comments_words"])
    comments = extract_social_emotion_value_1(pieces[4]["comments_words"])
    print(type(pieces[4]["comments_words"]))
    print(pieces[4]["label"])
    print(len(comments))
    print(comments)
    all = []
    for i in range(len(comments)):
        all = all+comments[i]
    str1 = " ".join(all)
    #print(str1)
cut_test = " ".join(jieba.cut(str1))
wordcloud = WordCloud(
    font_path="C:/Windows/Fonts/simsun.ttc",
    background_color="white",width = 1000,height=800,stopwords=stop_w).generate(str1)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
'''

