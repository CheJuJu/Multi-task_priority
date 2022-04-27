# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
import os

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    log_path = config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
           
            model.zero_grad()
            
            l1=MultiCEFocalLoss(class_num = 3,alpha=torch.tensor([0.3,0.3,0.4]))
            l2=MultiCEFocalLoss(class_num = 3,alpha=torch.tensor([0.05,0.15,0.8]))
            loss = 0.5*l1(outputs[0].cpu(), labels[0].cpu()) + 0.5*l2(outputs[1].cpu(), labels[1].cpu())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true1 = labels[0].data.cpu()
                true2 = labels[1].data.cpu()
                predic1 = torch.max(outputs[0].data, 1)[1].cpu()
                predic2 = torch.max(outputs[1].data, 1)[1].cpu()
                train_acc1 = metrics.accuracy_score(true1, predic1)
                train_acc2 = metrics.accuracy_score(true2, predic2)
                dev_acc1,dev_acc2, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc_pri: {2:>6.2%}, Train Acc_kind: {3:>6.2%},  Val Loss: {4:>5.2},  Val Acc_pri: {5:>6.2%}, Val Acc_kind: {6:>6.2%},  Time: {7} {8}'
                print(msg.format(total_batch, loss.item(), train_acc1, train_acc2, dev_loss, dev_acc1, dev_acc2, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter,log_path)


def test(config, model, test_iter,log_path):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc1,test_acc2, test_loss, test_report1,test_report2, test_confusion1,test_confusion2 = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc_pri: {1:>6.2%} ,Test Acc_kind: {2:>6.2%}'
    print(msg.format(test_loss, test_acc1,test_acc2))
    print("Precision, Recall and F1-Score...")
    print(test_report1)
    print(test_report2)
    print("Confusion Matrix...")
    print(test_confusion1)
    print(test_confusion2)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    if not os.path.exists(log_path):  # 如果路径不存在
        os.makedirs(log_path)
    file = open(log_path+'/report.txt','w')
    file.write(msg.format(test_loss, test_acc1,test_acc2)+'\n\n')
    file.write('Precision, Recall and F1-Score...\n')
    file.write(str(test_report1)+'\n')
    file.write(str(test_report2)+'\n')
    file.write('Confusion Matrix...\n')
    file.write(str(test_confusion1))
    file.write(str(test_confusion2))


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all1 = np.array([], dtype=int)
    labels_all1 = np.array([], dtype=int)
    predict_all2 = np.array([], dtype=int)
    labels_all2 = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            l1=MultiCEFocalLoss(class_num = 3,alpha=torch.tensor([0.3,0.3,0.4]))  
            l2=MultiCEFocalLoss(class_num = 3,alpha=torch.tensor([0.05,0.15,0.8]))
            loss = 0.5*l1(outputs[0].cpu(), labels[0].cpu()) + 0.5*l2(outputs[1].cpu(), labels[1].cpu())
            loss_total += loss

            labels1 = labels[0].data.cpu().numpy()
            predic1 = torch.max(outputs[0].data, 1)[1].cpu().numpy()
            labels_all1 = np.append(labels_all1, labels1)
            predict_all1 = np.append(predict_all1, predic1)

            labels2 = labels[1].data.cpu().numpy()
            predic2 = torch.max(outputs[1].data, 1)[1].cpu().numpy()
            labels_all2 = np.append(labels_all2, labels2)
            predict_all2 = np.append(predict_all2, predic2)

    acc1 = metrics.accuracy_score(labels_all1, predict_all1)
    acc2 = metrics.accuracy_score(labels_all2, predict_all2)
    if test:
        report1 = metrics.classification_report(labels_all1, predict_all1,  digits=4)
        report2 = metrics.classification_report(labels_all2, predict_all2,  digits=4)
        confusion1 = metrics.confusion_matrix(labels_all1, predict_all1)
        confusion2 = metrics.confusion_matrix(labels_all2, predict_all2)
        return acc1, acc2, loss_total / len(data_iter), report1, report2, confusion1 ,confusion2
    return acc1, acc2, loss_total / len(data_iter)
