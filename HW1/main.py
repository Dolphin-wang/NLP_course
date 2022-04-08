import os
import re
import ssl
import jieba
from collections import Counter
import math

def get_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1

def get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1

def get_trigram_tf(tf_dic, words):
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1


def cal_unigram(context,vocab,flag='char'):
    str1 = '字' if flag == 'char' else '词'
    token_num = len(context)
    entropy_1gram = sum([-(vocab[_]/token_num)*math.log((vocab[_]/token_num),2) for _ in vocab])
    print("基于"+str1+"的一元信息熵:", entropy_1gram)

def cal_bigram(corpus, words_tf, flag='char'):
    str1 = '字' if flag == 'char' else '词'
    words_len = len(corpus)
    bigram_len = words_len

    bigram_tf = {}
    get_bigram_tf(bigram_tf, corpus)
  
    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于"+str1+"的二元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

def cal_trigram(corpus, words_tf, flag='char'):
    str1 = '字' if flag == 'char' else '词'
    trigram_len = len(corpus)
    trigram_tf = {}

    get_bigram_tf(words_tf, corpus)
    get_trigram_tf(trigram_tf, corpus)

    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len  # 计算联合概率p(x,y)
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
    print("基于"+str1+"的三元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")


file_path_list = []
root_path = './dataset'
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?「」@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'


for file in os.listdir(root_path):
    file_path_list.append(os.path.join(root_path, file))
print(file_path_list)
corpus=[]
for file_path in file_path_list:
    with open(file_path,encoding="ANSI") as f:
        text = f.read()[40:]
        text = re.sub(r1, '', text)
        text = text.replace("\n", '')
        text = text.replace("\u3000", '')
        text = text.replace(" ", '')
        corpus += text
corpus="".join(corpus)
len_char = len(corpus)

text_character = corpus
text_word = jieba.lcut(corpus)

character_tf = Counter(text_character)#用于统计词频
word_tf= Counter(text_word)#用于统计词频

print("字语料库字数:", len(text_character))
print("不同字的个数:", len(character_tf))
print("不同词的个数:", len(word_tf))
print("分词个数:", len(text_word))
print("平均词长:", round(len(text_character) / len(text_word), 5))



cal_unigram(text_character, character_tf, flag = 'char')
cal_bigram(text_character, character_tf, flag = 'char')
cal_trigram(text_character, character_tf, flag = 'char')
cal_unigram(text_word, word_tf, flag = 'word')
cal_bigram(text_word, word_tf, flag = 'word')
cal_trigram(text_word, word_tf, flag = 'word')




