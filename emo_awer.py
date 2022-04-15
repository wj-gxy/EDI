# coding:utf-8
'''
co-attention cite dEFEND
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionLayer(nn.Module):
    def __init__(self, device, input_last=128, attention_dim=100):
        '''
            Attention layer as propsosed in paper.
        '''
        super(AttentionLayer, self).__init__()

        self.attention_dim = 100
        self.input_last = 128
        self.epsilon = torch.tensor([1e-07]).to(device)
        self.device = device
        # initialize parametres
        self.W = nn.Parameter(torch.Tensor((input_last, attention_dim)))
        self.b = nn.Parameter(torch.Tensor((attention_dim)))
        self.u = nn.Parameter(torch.Tensor((attention_dim, 1)))

        # register params
        self.register_parameter("W", self.W)
        self.register_parameter("b", self.b)
        self.register_parameter("u", self.u)

        # initialize param data
        self.W.data = torch.randn((input_last, attention_dim))
        self.b.data = torch.randn((attention_dim))
        self.u.data = torch.randn((attention_dim, 1))

    def forward(self, x):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)

        uit = torch.tanh(torch.matmul(x, self.W) + self.b)
        ait = torch.matmul(uit, self.u)
        ait = torch.squeeze(ait, -1)
        ait = torch.exp(ait)

        ait = ait / (torch.sum(ait, dim=1, keepdims=True) + self.epsilon).to(self.device)
        ait = torch.unsqueeze(ait, -1)
        weighted_input = x * ait
        output = torch.sum(weighted_input, dim=1)

        return output,ait


class CoAttention(nn.Module):
    def __init__(self, device, latent_dim=128):
        super(CoAttention, self).__init__()

        self.latent_dim = latent_dim
        self.k = 80
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))

        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Ws = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))

        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))

        # register weights and bias as params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)

        # initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Ws.data = torch.randn((self.k, self.latent_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))

    def forward(self, sentence_rep, comment_rep):
        sentence_rep_trans = sentence_rep.transpose(2, 1)
        comment_rep_trans = comment_rep.transpose(2, 1)

        L = torch.tanh(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_trans))

        L_trans = L.transpose(2, 1)

        Hs = torch.tanh(
            torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))

        Hc = torch.tanh(
            torch.matmul(self.Wc, comment_rep_trans) + torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))

        As = F.softmax(torch.matmul(self.whs, Hs), dim=2)

        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=2)

        As = As.transpose(2, 1)

        Ac = Ac.transpose(2, 1)

        # sentence_rep_trans = sentence_rep_trans.transpose(0, 2)
        # comment_rep_trans = comment_rep_trans.transpose(0, 2)

        co_s = torch.matmul(sentence_rep_trans, As)

        co_c = torch.matmul(comment_rep_trans, Ac)

        co_sc = torch.cat([co_s, co_c], dim=1)

        return torch.squeeze(co_sc, -1), As, Ac


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class BACCA(nn.Module):
    def __init__(self, config, word_emb, emo_emb):

        super(BACCA, self).__init__()

        self.embedding_word = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.embedding_emo = nn.Embedding.from_pretrained(torch.Tensor(emo_emb))
        self.word_dim = self.embedding_word.embedding_dim

        self.attention = AttentionLayer(config.device)
        self.conv1 = nn.Conv1d(self.word_dim, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv1d(self.word_dim, out_channels=64, kernel_size=3)
        self.pool = GlobalMaxPool1d()
        self.coattention = CoAttention(config.device, 128)
        self.fc = nn.Linear(384, 2)
        self.softamx = nn.Softmax(dim=1)
        # coattention weight
        self.latent_dim = 128
        self.k = 80
        self.Wl = Variable(torch.rand((self.latent_dim, self.latent_dim), requires_grad=True).to(config.device))

        self.Wc = Variable(torch.rand((self.k, self.latent_dim), requires_grad=True).to(config.device))
        self.Ws = Variable(torch.rand((self.k, self.latent_dim), requires_grad=True).to(config.device))

        self.whs = Variable(torch.rand((1, self.k), requires_grad=True).to(config.device))
        self.whc = Variable(torch.rand((1, self.k), requires_grad=True).to(config.device))

    def forward(self, content, comments):
        global x3_max
        embedded_content = self.embedding_word(content)
        embedded_content_emo = self.embedding_emo(content)
        embedded_comment_emo = self.embedding_emo(comments)
        embedded_content = embedded_content.transpose(1, 0)
        embedded_content_emo = embedded_content_emo.transpose(1, 0)
        embedded_comment_emo = embedded_comment_emo.transpose(1, 0)
        x3_list = []
        x4_list = []
        for sentence in embedded_content_emo:
            sentence = sentence.permute(0, 2, 1)

            x3 = F.relu(self.conv1(sentence))
            x3_max = self.pool(x3).squeeze(-1)
            x3_2 = F.relu(self.conv2(sentence))
            x3_max_2 = self.pool(x3_2).squeeze(-1)

            x3_emo = torch.cat([x3_max, x3_max_2], dim=1)

            x3_list.append(x3_emo)

        for comment in embedded_comment_emo:
            comment = comment.permute(0, 2, 1)
            x4 = F.relu(self.conv1(comment))
            x4_max = self.pool(x4).squeeze(-1)
            x4_2 = F.relu(self.conv2(comment))
            x4_max_2 = self.pool(x4_2).squeeze(-1)
            x4_emo = torch.cat([x4_max, x4_max_2], dim=1)
            x4_list.append(x4_emo)
        xb = torch.stack(x3_list)
        xd = torch.stack(x4_list)

        xb = xb.transpose(1, 0)
        xd = xd.transpose(1, 0)
        xd_a,ait = self.attention(xd)
        #print(xd_a.size())
        coatten_emo, A_S, A_C = self.coattention(xb, xd)
        co = torch.cat([xd_a, coatten_emo], dim=1)

        preds = self.fc(co)

        preds = self.softamx(preds)

        return preds, A_S, A_C, ait

    def initHidden(self):
        word_lstm_weight = Variable(torch.zeros(2, config.batch_size, self.embedding_dim).to(config.device))
        comment_lstm_weight = Variable(torch.zeros(2, config.batch_size, self.embedding_dim).to(config.device))
        content_lstm_weight = Variable(torch.zeros(2, config.batch_size, self.embedding_dim).to(config.device))

        return (word_lstm_weight, comment_lstm_weight, content_lstm_weight)


if __name__ == "__main__":
    import os
    import numpy as np
    import time
    from config import Config

    config = Config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    content = torch.rand(12, 1, 32).type(torch.LongTensor).to(device)
    comment = torch.rand(12, 10, 32).type(torch.LongTensor).to(device)
    emotion = torch.rand(12, 47).type(torch.LongTensor).to(device)

    word_emb = np.random.randn(84574, 300)
    emo_emb = np.random.randn(10000, 300)
    defend = BACCA(config, word_emb, emo_emb).to(device)

    defend = defend.to(device)

    since = time.time()
    pred = defend(content, comment)

    print(f"total time: {time.time() - since}")
    # print(f"out shape: {out.shape()}")
