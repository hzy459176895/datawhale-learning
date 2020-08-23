

"""
nlp常见特征提取方法
"""


import math

# TF_IDF_weight函数功能：传入corpus，返回其特征向量值向量
# 输入的corpus格式:["我 来到 中国 旅游", "中国 欢迎 你"，"我 喜欢 来到 中国 天安门"]
def TF_IDF_weight(corpus):

    # 词库的构建
    weight_long = [eve.split() for eve in corpus]
    word_all = []
    for eve in weight_long:
        for x in eve:
            if len(x) > 1:
                word_all.append(x)
    word_all = list(set(word_all))  # 集合去重词库
    # 开始计算tf-idf
    weight = [[] for i in corpus]
    weight_idf = [[] for i in corpus]
    for word in word_all:
        for i in range(len(corpus)):
            temp_list = corpus[i].split()
            n1 = temp_list.count(word)
            n2 = len(temp_list)
            tf = n1/n2
            n3 = len(corpus)
            n4 = 0
            for eve in corpus:
                temp_list_ = eve.split()
                if word in temp_list_:
                    n4 += 1
            idf = math.log(((n3+1)/(n4+1))+1,10)
            weight_idf[i].append(idf)
            weight[i].append(tf*idf)

    # L2范式归一化过程
    l2_weight = [[] for i in range(len(corpus))]
    for e in range(len(weight)):
        all2plus = 0
        for ev in weight[e]:
            all2plus += ev**2
        for ev in weight[e]:
            l2_weight[e].append(ev/(all2plus**0.5))
    return l2_weight  # 返回最终结果


corpus = ["我 来到 中国 旅游", "中国 欢迎 你", "我 喜欢 来到 中国 天安门"]
result_list = TF_IDF_weight(corpus)
for weight in result_list:
    print(weight)

# sklearn中的tf-idf特征提取
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()  # 实例化
transformer = TfidfTransformer()
corpus = ["我 来到 中国 旅游", "中国 欢迎 你","我 喜欢 来到 中国 天安门"]
result_list2 = transformer.fit_transform(vectorizer.fit_transform(corpus)).toarray().tolist()
for weight in result_list2:
    print(weight)


