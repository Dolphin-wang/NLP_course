from __future__ import print_function
import jieba
import argparse
import numpy as np
import random
import sys
import os
import pickle
import re

# from tensorflow.python.keras.callbacks import TensorBoard
# from tensorflow.python.keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, LSTM, Embedding
from sklearn.model_selection import learning_curve
EMBEDDING_DIM = 300
def load_vectors(fname):
    fin = open(fname, "r",encoding='utf-8')
    w2v = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        w2v[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return w2v


def read_dataset(maxlen=40, step=3):
    print('preparing datasets...')
    path = 'datasets'
    files = os.listdir(path)
    text = ''  # 直接拼接的所有语聊内容
    for file in files:
        if not os.path.isdir(file):
            text += open(os.path.join(path, file), 'r', encoding="ANSI").read().strip()

    text = text[100:]
    text = text[0:len(text)]
    text = text.replace("\u3000", '')
    text = text.replace(" ", '')
    text_words = jieba.lcut(text)  # 将原语聊转为words

    print('total words:', len(text_words))
    sentences = []  # 句子
    next_words = []  # 句子的下一个字符
    for i in range(0, len(text_words) - maxlen, step):
        sentences.append(" ".join(text_words[i: i + maxlen]))  # 每个sentence是带空格符的一句话
        next_words.append(text_words[i + maxlen])  # 每个next_words一个单独的词
    print('nb sentences:', len(sentences))

    print('tokenizing...')
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(text_words)
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    X = tokenizer.texts_to_sequences(sentences)
    y = tokenizer.texts_to_matrix(next_words)
    X = pad_sequences(X, maxlen=maxlen)
    y = np.array(y,dtype="float32")
    print('vocab size:', len(tokenizer.word_index))
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)

    if os.path.exists('embedding_matrix.dat'):
        print('building embedding_matrix...')
        embedding_matrix = pickle.load(open('embedding_matrix.dat', 'rb'))
    else:
        print('loading word vectors...')
        word_vec = load_vectors('word2Vec.txt')
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        print('building embedding_matrix...')
        for word, i in word_index.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
            else:
                embedding_matrix[word_index[word]] = 1
        pickle.dump(embedding_matrix, open('embedding_matrix.dat', 'wb'))

    return tokenizer, index_word, embedding_matrix, text_words, X, y


class Generator:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 20
        self.STEP = 3
        self.ITERATION = 500
        self.tokenizer, self.index_word, self.embedding_matrix, self.text_words, self.X, self.y = \
            read_dataset(maxlen=self.MAX_SEQUENCE_LENGTH, step=self.STEP)

        if os.path.exists('saved_model.h5'):
            print('loading saved model...')
            self.model = load_model('saved_model.h5')
        else:

            print('Build model...')
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
            print("len:",len(self.tokenizer.word_index))
            x = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                          output_dim=EMBEDDING_DIM,
                          input_length=self.MAX_SEQUENCE_LENGTH,
                          weights=[self.embedding_matrix],
                          trainable=False,
                          dtype='float32')(inputs)

            # x = Bidirectional(LSTM(600, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(x)
            # x = LSTM(600, dropout=0.2, recurrent_dropout=0.1)(x)
            # 可以选择把上面两行注释掉，选择下面这行，简化模型，加快训练的速度
            x = LSTM(600, dropout=0.2, recurrent_dropout=0.1)(x)
            x = Dense(len(self.tokenizer.word_index) + 1)(x)
            predictions = Activation('softmax')(x)
            model = Model(inputs, predictions)
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='adam')
            # plot_model(model, to_file='model.png')
            self.model = model

    @staticmethod
    def sample(preds, temperature=1.0):

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def train(self):
        """"
        训练模型，在每一次迭代后输出生成的文本
        """
        # tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        for i in range(1, self.ITERATION):
            print()
            print('-' * 50)
            print('Iteration', i)
            self.model.fit(self.X, self.y, batch_size=128)  # 训练

            self.model.save('saved_model.h5')
            print('model saved')
            self.inference()

    def inference(self, seed=None):
        if seed is None:
            # 随机生成种子文本
            start_index = random.randint(0, len(self.text_words) - self.MAX_SEQUENCE_LENGTH - 1)
            seed_words = self.text_words[start_index: start_index + self.MAX_SEQUENCE_LENGTH]
            seed = "".join(seed_words)
        else:
            seed_words = jieba.lcut(seed)

        for diversity in [0.5,1.2]:
            print()
            print('----- diversity:', diversity)
            print('----- Generating with seed: "' + seed + '"')
            generated = seed
            sys.stdout.write(generated)  # 打印文本

            x = self.tokenizer.texts_to_sequences([" ".join(seed_words)])
            x = pad_sequences(x, maxlen=self.MAX_SEQUENCE_LENGTH)
            for i in range(100):  # 连续生成后续words
                preds = self.model.predict(x, verbose=0)[0]  # 预测下一个结果
                next_index = self.sample(preds, diversity)  # 抽样出下一个字符的索引值
                next_word = self.index_word[next_index]  # 检出下一个字符

                generated += next_word
                x = np.delete(x, 0, -1)
                x = np.append(x, [[next_index]], axis=1)  # 输入后移一个word

                sys.stdout.write(next_word)  # 连续打印
                sys.stdout.flush()  # 刷新控制台
            print()


if __name__ == '__main__':
    
    gen = Generator()
    # gen.train()
    gen.inference(seed="这日傍晚，令狐冲又在崖上凝目眺望，却见两个人形迅速异常的走上崖来，前面一人衣裙飘飘，是个女子。")

        