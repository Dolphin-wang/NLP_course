# 平均信息熵作业
## 数据预处理
本阶段用于对16本小说文本数据进行预处理，具体内容为
1、去除数据集中前几行的广告等无意义文本
2、去除数据集中一些特殊符号等无效信息
3、将16本小说进行整合，形成一个语料库
```python
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
```

## 分词
本阶段用于对阶段1所输出的语料库进行使用jieba分词，分别形成以字为级别的语料库和以为级别的语料库

字语料库
```python
text_character = corpus
```
词语料库
```python
text_word = jieba.lcut(corpus)
```

## 统计词频并进行计算
统计词频
```python
ct_character = Counter(text_character)#用于统计词频
vocab_character = ct_character.most_common()

ct_word= Counter(text_word)#用于统计词频
vocab_word= ct_word.most_common()
```
计算平均信息熵
```python
def cal_entropy(context,vocab1):
    token_num = len(context)
    entropy_1gram = sum([-(eve[1]/token_num)*math.log((eve[1]/token_num),2) for eve in vocab1])
    print("词库总词数：", token_num, " ", "不同词的个数：", len(vocab1))
    print("出现频率前5的1-gram词语：", vocab1[:20])
    print("entropy_1gram:", entropy_1gram)
```
结果
