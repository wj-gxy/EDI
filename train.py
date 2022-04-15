import os
import time
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from defend import dEFENDNet
from config import Config
from emo_awer import BACCA
#from cnn import BACCA
from utils import date, MPCNDataset, load_embedding, get_time_dif, load_embeddings
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from utils_weibo import date, MPCNDataset, load_embedding, get_time_dif, load_embeddings
def train(train_dataloader, valid_dataloader, model, config, model_path):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    total_batch = 0  # 记录进行了多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很多次没有效果提升
    for epoch in range(config.train_epochs):
        print('Epoch[{}/{}]'.format(epoch + 1, config.train_epochs))
        for batch in train_dataloader:
            user_reviews, item_reviews, ratings = [x.to(config.device) for x in batch]
            outputs, As, Ac, ait = model(user_reviews, item_reviews)
            model.zero_grad()
            loss = F.cross_entropy(outputs, ratings.long())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = ratings.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(model, valid_dataloader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model, model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter:{0:>6},Train Loss{1:>5.2},Train Acc{2:>6.2},Val Loss{3:>5.2},Val Acc:{4:>6.2%},Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvment:
                print('再检验数据集上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break
            if flag:
                break


def evaluate(model, dev_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in dev_iter:
            user_reviews, item_reviews, ratings = [x.to(config.device) for x in batch]
            outputs, As, Ac, ait = model(user_reviews, item_reviews)
            loss = F.cross_entropy(outputs, ratings.long())
            loss = loss.item()
            loss_total += loss
            labels = ratings.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = accuracy_score(labels_all, predict_all)
    if test:
        report = classification_report(labels_all, predict_all, target_names=config.class_list, digits=3)
        confusion = confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion
    return acc, loss_total / len(dev_iter)


def test(test_iter, model, config):
    torch.load(config.saved_model)  # 加载模型
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    # print(test_acc,test_loss)
    print('Precision, Recall and F1-Score')
    print(test_report)
    print('Confusion Maxtrix')
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print('使用时间:', time_dif)


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load word2vec and data...')
    word_emb, word_dict = load_embedding(config.emotion_file)
    emo_emb, emo_dict = load_embedding(config.emotion_file)

    # Train
    train_dataset = MPCNDataset(config.train_file, word_dict, emo_dict, config)
    valid_dataset = MPCNDataset(config.valid_file, word_dict, emo_dict, config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    os.makedirs(os.path.dirname(config.saved_model), exist_ok=True)  # make dir if it isn't exist.
    model = BACCA(config, word_emb, emo_emb).to(config.device)
    train(train_dlr, valid_dlr, model, config, config.saved_model)

    # Test
    test_dataset = MPCNDataset(config.test_file, word_dict, emo_dict, config)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)
    test(test_dlr, torch.load(config.saved_model), config)
