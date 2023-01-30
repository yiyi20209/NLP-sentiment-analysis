from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
import numpy as np
from torchtext.data.utils import get_tokenizer

Train_Ro_path = './Dataset/train_ro.txt'


def trainw():
    # 训练词向量生成网络
    model = Word2Vec(
        LineSentence(open(Train_Ro_path, 'r')),
        sg=0,
        window=3,
        min_count=30,
        workers=8
    )
    # model = Word2Vec()

    # 词向量保存
    model.wv.save_word2vec_format('./Dataset/data.vector', binary=False)

    # 模型保存
    model.save('./Dataset/test.model')


def loadw():
    # 1 通过模型加载词向量(recommend)
    w2v_model = gensim.models.Word2Vec.load('test.model')

    dic = w2v_model.wv.index_to_key
    print(dic)
    print(len(dic))

    print(w2v_model.wv['you'])
    print(w2v_model.wv['you'].shape)

    print('badapple' in dic)
    print('you' in dic)
    # print(model.most_similar('you', topn=1))


trainw()
# loadw()
# tokenizer = get_tokenizer('basic_english')
#
#
# def build_sentence_vector(sentence, w2v_model, size=100):
#     sen_vec = np.zeros((size,))
#     count = 0
#     for word in tokenizer(sentence):
#         try:
#             sen_vec += w2v_model.wv[word]  # .reshape((1,size))
#             count += 1
#             print(word)
#         except KeyError:
#             continue
#     if count != 0:
#         sen_vec /= count
#     return sen_vec


# s1v = build_sentence_vector('I love apple apple apple apple apple apple', gensim.models.Word2Vec.load('test.model'), 100)
# print(s1v)
# print(s1v.shape)

# 对每一个句子预处理完后使用build_sentence_vector()即可获得每一个句子的嵌入向量