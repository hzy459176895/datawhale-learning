
"""
逻辑回归鸢尾花数据案例
"""

# 基础函数库
import numpy as np
import pandas as pd
# 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 我们利用 sklearn 中自带的 iris 数据作为数据载入，并利用Pandas转化为DataFrame格式
from sklearn.datasets import load_iris
data = load_iris()  # 得到数据特征
iris_target = data.target  # 得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)  # 利用Pandas转化为 DataFrame格式
# df.info() 查看数据详细信息
print('iris数据集情况.....')
print(iris_features.info())
# 查看前面和尾部一些数据案例
print('前面一些数据例子...')
print(iris_features.head())
print('后面一些数据例子...')
print(iris_features.tail())


# 特征值统计情况
print(iris_features.describe())
# 查看类别的情况
print(pd.Series(iris_target).value_counts())



# 合并标签和特征信息
iris_all = iris_features.copy()  # 进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target
# 数据和标签组合可视化（可以看到对角线上是各个属性的直方图（分布图），而非对角线上是两个不同属性之间的相关图）
sns.pairplot(data=iris_all, diag_kind='hist', hue='target')
plt.show()

# 不同特征列对目标数据的箱型图规律
for col in iris_features.columns:
 sns.boxplot(x='target', y=col, saturation=0.5,palette='pastel', data=iris_all)
 plt.title(col)
 plt.show()


# 选取其前三个特征绘制三维散点图
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
iris_all_class0 = iris_all[iris_all['target']==0].values
iris_all_class1 = iris_all[iris_all['target']==1].values
iris_all_class2 = iris_all[iris_all['target']==2].values

ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')
ax.scatter(iris_all_class1[:,0], iris_all_class1[:,1],iris_all_class1[:,2],label='versicolor')
ax.scatter(iris_all_class2[:,0], iris_all_class2[:,1],iris_all_class2[:,2],label='virginica')
plt.legend()
plt.show()

# # === 逻辑回归训练和验证 =====================================
# from sklearn.model_selection import train_test_split
# ## 选择其类别为0和1的样本 （不包括类别为2的样本） --- 这里做二分类测试
# iris_features_part = iris_features.iloc[:100]
# iris_target_part = iris_target[:100]
# ## 测试集大小为20%， 80%/20%分
# x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part,
# test_size = 0.2, random_state = 2020)
# ## 从sklearn中导入逻辑回归模型
# from sklearn.linear_model import LogisticRegression
# ## 定义 逻辑回归模型
# clf = LogisticRegression(random_state=0, solver='lbfgs')
# # 在训练集上训练逻辑回归模型
# clf.fit(x_train, y_train)
#
# ## 查看其对应的w
# print('the weight of Logistic Regression:',clf.coef_)
# ## 查看其对应的w0
# print('the intercept(w0) of Logistic Regression:',clf.intercept_)
#
# ## 在训练集和测试集上分布利用训练好的模型进行预测
# train_predict = clf.predict(x_train)
# test_predict = clf.predict(x_test)
# from sklearn import metrics
# ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
# print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
# print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))
# ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
# confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
# print('The confusion matrix result:\n',confusion_matrix_result)
# # 利用热力图对于结果进行可视化
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.show()


# === 就三分类的例子 ==============
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 测试集大小为20%， 80%/20%分
x_train, x_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size =
0.2, random_state = 2020)
# 定义 逻辑回归模型
clf = LogisticRegression(random_state=0, solver='lbfgs')
# 在训练集上训练逻辑回归模型
clf.fit(x_train, y_train)
# 查看其对应的w
print('----------------------------')
print('the weight of Logistic Regression:\n',clf.coef_)
# 查看其对应的w0
print('the intercept(w0) of Logistic Regression:\n',clf.intercept_)

from sklearn import metrics
## 由于这个是3分类，所有我们这里得到了三个逻辑回归模型的参数，其三个逻辑回归组合起来即可实现三分类。
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

## 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所有我们可以利用 predict_proba 函数预测其概率
train_predict_proba = clf.predict_proba(x_train)
test_predict_proba = clf.predict_proba(x_test)

print('The test predict Probability of each class:\n',test_predict_proba)
## 其中第一列代表预测为0类的概率，第二列代表预测为1类的概率，第三列代表预测为2类的概率。
## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

## 查看混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)
# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()