import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
from torch.utils.data import TensorDataset,DataLoader
from torch.optim.lr_scheduler import *

Word_Vector_path = './Dataset/data.vector'
Train_Ro_path = './Dataset/train_ro.txt'
Valid_Ro_path = './Dataset/valid_ro.txt'

learning_rate = 0.001  # 学习率
BATCH_SIZE = 64  # 训练批量
EPOCHS = 5  # 训练轮数
model_path = None  # 预训练模型路径


def build_word2id(file, save_to_path=None):
    """
    :param file: word2id保存地址
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = ['./Dataset/train.txt', './Dataset/validation.txt']

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    if save_to_path:
        with open(file, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w + '\t')
                f.write(str(word2id[w]))
                f.write('\n')

    return word2id


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs



def cat_to_id(classes=None):
    """
    :param classes: 分类标签
    :return: {分类标签：id}
    """
    if not classes:
        classes = ['0', '1', '2']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id



def load_corpus(path, word2id, max_sen_len=50):
    """
    :param path: 样本语料库的文件
    :return: 文本内容contents，以及分类标签labels(onehot形式)
    """
    _, cat2id = cat_to_id()
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [word2id.get(w, 0) for w in sp[1:]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word2id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('Total sample num：%d' % (len(labels)))
    print('class num：')
    for w in counter:
        print(w, counter[w])

    contents = np.asarray(contents)
    labels = np.array([cat2id[l] for l in labels])

    return contents, labels


word2id = build_word2id('./Dataset/word2id.txt')
# print(word2id)
word2vec = build_word2vec(Word_Vector_path, word2id)
print(word2vec.shape)


class CONFIG():
    update_w2v = True  # 是否在训练中更新w2v
    vocab_size = word2vec.shape[0]  # 词汇量，与word2id中的词汇量一致
    n_class = 3  # 分类数：分别为pos和neg
    embedding_dim = 100  # 词向量维度
    drop_keep_prob = 0.5  # dropout层，参数keep的比例
    kernel_num = 64  # 卷积层filter的数量
    kernel_size = [3, 4, 5]  # 卷积核的尺寸
    pretrained_embed = word2vec  # 预训练的词嵌入模型


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        n_class = config.n_class
        embedding_dim = config.embedding_dim
        kernel_num = config.kernel_num
        kernel_size = config.kernel_size
        drop_keep_prob = config.drop_keep_prob
        pretrained_embed = config.pretrained_embed

        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embedding.weight.requires_grad = update_w2v
        # 卷积层
        self.conv1 = nn.Conv2d(1, kernel_num, (kernel_size[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, kernel_num, (kernel_size[1], embedding_dim))
        self.conv3 = nn.Conv2d(1, kernel_num, (kernel_size[2], embedding_dim))
        # Dropout
        self.dropout = nn.Dropout(drop_keep_prob)
        # 全连接层
        self.fc = nn.Linear(len(kernel_size) * kernel_num, n_class)

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv2)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv3)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



def train(dataloader, epoch):
    # 定义训练过程
    train_loss, train_acc = 0.0, 0.0
    count, correct = 0, 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

        if (batch_idx + 1) % 100 == 0:
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))

    train_loss *= BATCH_SIZE
    train_loss /= len(dataloader.dataset)
    train_acc = correct / count
    print('\ntrain epoch: {}\taverage loss: {:.6f}\taccuracy:{:.4f}%\n'.format(epoch, train_loss, 100. * train_acc))
    scheduler.step()

    return train_loss, train_acc


def validation(dataloader, epoch):
    model.eval()
    # 验证过程
    val_loss, val_acc = 0.0, 0.0
    count, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

    val_loss *= BATCH_SIZE
    val_loss /= len(dataloader.dataset)
    val_acc = correct / count
    # 打印准确率
    print(
        'validation:train epoch: {}\taverage loss: {:.6f}\t accuracy:{:.2f}%\n'.format(epoch, val_loss, 100 * val_acc))

    return val_loss, val_acc


if __name__ == '__main__':

    # assert word2vec.shape == (58954, 50)
    # print(word2vec)
    print('train set: ')
    train_contents, train_labels = load_corpus(Train_Ro_path, word2id, max_sen_len=100)
    print('\nvalidation set: ')
    val_contents, val_labels = load_corpus(Valid_Ro_path, word2id, max_sen_len=100)
    # print('\ntest set: ')
    # test_contents, test_labels = load_corpus('./Dataset/test.txt', word2id, max_sen_len=50)

    config = CONFIG()  # 配置模型参数

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.float),
                                  torch.from_numpy(train_labels).type(torch.long))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2)

    val_dataset = TensorDataset(torch.from_numpy(val_contents).type(torch.float),
                                torch.from_numpy(val_labels).type(torch.long))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)

    # 配置模型，是否继续上一次的训练
    model = TextCNN(config)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5)

    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(train_dataloader, epoch)
        val_loss, val_acc = validation(val_dataloader, epoch)
        train_losses.append(tr_loss)
        train_acces.append(tr_acc)
        val_losses.append(val_loss)
        val_acces.append(val_acc)

    model_pth = 'model_' + str(time.time()) + '.pth'
    torch.save(model.state_dict(), model_pth)



