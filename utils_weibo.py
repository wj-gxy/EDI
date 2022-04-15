import os
import time
from datetime import timedelta
import json
import torch
from torch.utils.data import Dataset
from emotion_ch_1 import extract_social_emotion_value_1
from config import Config


def load_embedding(word2vec_file):
    with open(word2vec_file, 'r', encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        line1 = f.readlines()
        line1 = [l.strip() for l in line1]
        lines = line1[1:]
        for line in lines:
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


def load_embeddings(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])

            word_dict[tokens[0]] = len(word_dict) + 1
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def get_time_dif(start_time):
    '''
    获取已经使用时间
    :param start_time:
    :return:
    '''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(time_dif))


class MPCNDataset(Dataset):
    def __init__(self, data_path, word_dict, emotion_dict, config):
        self.content_count = config.content_count
        self.content_length = config.content_length
        self.comment_count = config.comment_count

        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item
        self.review_length = config.review_length
        self.PAD_idx = word_dict[config.PAD_WORD]
        self.emo_idx = emotion_dict[config.PAD_WORD]
        label2idx = {'fake': 0, 'real': 1}
        contents = []

        comments = []
        label = []
        split_dataset = [json.load(open(os.path.join(data_path), 'r', encoding='utf-8'))]  # 打开.json的文件
        split_dataset = dict(zip(['twitter15'], split_dataset))
        for p in split_dataset['twitter15']:
            contents.append([p['content_words']])
            label.append(label2idx[p['label']])

            comments.append(p['comments_words'])

        reviews = []
        comment = []
        for i in range(len(comments)):
            comment.append(extract_social_emotion_value_1(comments[i]))
        for i in range(len(comment)):
            b = []
            for j in range(len(comment[i])):
                a = []
                for w in str(comment[i][j]):
                    a.append(word_dict.get(w, self.PAD_idx))
                b.append(a)
            reviews.append(b)
        content = []
        for i in range(len(contents)):
            b = []
            for j in range(len(contents[i])):
                a = []
                for w in str(contents[i][j]):
                    a.append(word_dict.get(w, self.PAD_idx))
                b.append(a)
            content.append(b)

        user_post, user_commetns = self._get_content_reviews(content, reviews)  # Gather reviews for user
        retain_idx = [idx for idx in range(user_post.shape[0])]
        self.user_post = user_post[retain_idx]
        self.user_comments = user_commetns[retain_idx]
        self.rating = torch.Tensor(label)[retain_idx]

    def __getitem__(self, idx):
        return self.user_post[idx], self.user_comments[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_content_reviews(self, content, reviews):
        group_reviews = []
        group_content = []

        for i in range(len(reviews)):
            pad_reviews = self._pad_reviews(reviews[i])
            pad_content = self._pad_contetn((content[i]))
            group_reviews.append(pad_reviews)
            group_content.append(pad_content)
        return torch.LongTensor(group_content), torch.LongTensor(group_reviews)

    def _pad_reviews(self, reviews):
        count, length = self.comment_count, self.review_length

        reviews = reviews[:count] + [[self.emo_idx] * length] * (count - len(reviews))  # Certain count.
        reviews = [r[:length] + [0] * (length - len(r)) for r in reviews]  # Certain length of review.
        return reviews

    def _pad_contetn(self, content):
        # print(content)
        content_count, content_length = self.content_count, self.content_length
        reviews = content[:content_count] + [[self.PAD_idx] * content_length] * (
                    content_count - len(content))  # Certain count.
        reviews = [r[:content_length] + [0] * (content_length - len(r)) for r in reviews]  # Certain length of review.
        return reviews


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load word2vec and data...')
    word_emb, word_dict = load_embedding(config.word2vec_file)
    print(len(word_emb))
    print(len(word_dict))
    emo_emb, emo_dict = load_embedding(config.emotion_file)
    print(len(emo_emb))
    print(len(emo_dict))
    train_dataset = MPCNDataset(config.train_file, word_dict, emo_dict, config)
