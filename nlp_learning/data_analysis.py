
"""
数据初探
"""

import sys

''' 读取训练集，空格符分割，前20行 '''
# import pandas as pd
# train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=20)
# print(train_df)


''' 将text进行空格分割开，统计每一行的字符数量，给到text_len列 '''
# import pandas as pd
# train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=10000)
# # print(train_df.shape)  # 行/列数
# train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
# # print(train_df.head(20))

''' 查看text_len列的详情统计描述'''
# print(train_df['text_len'].describe())


# ''' 每一个文本长度的数据直方图统计 '''
# import matplotlib.pyplot as plt
# _ = plt.hist(train_df['text_len'], bins=200)
# plt.xlabel('Text char count')
# plt.title("Histogram of char count")
# plt.show()  # 显示图片


# ''' 所有类别统计情况 '''
# import matplotlib.pyplot as plt
# train_df['label'].value_counts().plot(kind='bar')
# plt.title('News class count')
# plt.xlabel("category")
# plt.show()  # 显示图片


''' 统计所有训练集中 所有词，出现次数最多的词，出现次数最少的词 '''
# from collections import Counter
# all_lines = ' '.join(list(train_df['text']))  # 将每一行text再以空格，连起来
# word_count = Counter(all_lines.split(" "))  # 统计数字和出现次数
# # 每一个字符和它出现的次数，组成元组，按照由多到少排序
# word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
# print(len(word_count))  # 总共字符类别有多少种
# print(word_count[0])  # 统计最多的字符类别
# print(word_count[-1])  # 统计最少的字符类别


''' 拓展1：假设字符3750，字符900和字符648是句子的标点符号，请分析赛题每篇新闻平均由多少个句子构成？ '''
# # # 1 统计没有标点的字符数 ####################
# # import pandas as pd
# # train_df = pd.read_csv('data/train_set.csv', sep='\t')
# # bioadian = ['3750', '900', '648']
# # train_df['text_len'] = train_df['text'].apply(lambda x: len([i for i in x.split(' ') if i not in bioadian]))
# # print(train_df['text_len'].describe())
#
# # 统计没有标点的句子数 (按照标点进行split，剩下的就是句子) ####################
# import re
# import pandas as pd
# train_df = pd.read_csv('data/train_set.csv', sep='\t')
# train_df['juzi_len'] = train_df['text'].apply(lambda x: len(re.split('3750|900|648', x)))
# print(train_df['juzi_len'].describe())


''' 统计每类新闻中出现次数最多的字符 '''
import pandas as pd
from collections import Counter
train_df = pd.read_csv('data/train_set.csv', sep='\t')
# 同一类的拼接到一起
for i in range(0, 14):
    df = train_df[train_df['label'] == i]['text']
    bioadian = ['3750', '900', '648']
    df_2 = df.apply(lambda x: [i for i in x.split(' ') if i not in bioadian])
    all_lines = str(df_2.values.tolist())
    word_count = Counter(all_lines.split(" "))  # 统计数字和出现次数
    word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)  # 排序
    print(i, word_count[0])  # 新闻类，次数最多的字符及次数
