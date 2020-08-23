
"""
逻辑回归测试demo
"""

# 基础函数库
import numpy as np
# 导入画图库
import matplotlib.pyplot as plt
import seaborn as sns
# 导入逻辑回归模型函数
from sklearn.linear_model import LogisticRegression
import sys


# 构造数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
print('数据：------')
print(x_fearures)
y_label = np.array([0, 0, 0, 1, 1, 1])
print('标签：------')
print(y_label)
# 查看数据散点图
plt.figure()
plt.scatter(x_fearures[:, 0], x_fearures[:, 1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
plt.show()

# 调用逻辑回归模型
lr_clf = LogisticRegression()
# 用逻辑回归模型拟合构造的数据集
lr_clf = lr_clf.fit(x_fearures, y_label)  # 其拟合方程为 y = w0 + w1 * x1 + w2 * x2
# 查看系数： 权重w1，w2和截距 w0
## 查看其对应模型的w
print('the weight of Logistic Regression:', lr_clf.coef_)
## 查看其对应模型的w0
print('the intercept(w0) of Logistic Regression:', lr_clf.intercept_)

# 可视化决策边界
plt.figure()
plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
nx, ny = 200, 100
# 获取feature中x轴和y轴上最小最大值
x_min, x_max = plt.xlim()  # 设定坐标范围
y_min, y_max = plt.ylim()
# np.linspace：创建等差数列，np.meshgrid：网格坐标矩阵
print(np.linspace(x_min, x_max, nx))  # 区间内创建200个等差数列的数
print('  ')
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
print(x_grid)
print(y_grid)
# np.c_: 添加列
z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
z_proba = z_proba[:, 1].reshape(x_grid.shape)
print(z_proba)
# plt.contour: 绘制轮廓
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')
plt.show()

# 添加两颗新样本并可视化
plt.figure()
# new point 1
x_fearures_new1 = np.array([[0, -1]])
plt.scatter(x_fearures_new1[:,0],x_fearures_new1[:,1], s=50, cmap='viridis')
# 注：plt.annotate(): 主要是用于在图中添加标注
plt.annotate(s='New point 1',xy=(0,-1),xytext= (-2,0),color='blue',
             arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))
# new point 2
x_fearures_new2 = np.array([[1, 2]])
plt.scatter(x_fearures_new2[:,0],x_fearures_new2[:,1], s=50, cmap='viridis')
plt.annotate(s='New point 2',xy=(1,2),xytext=(-1.5,2.5),color='red',
             arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))
# 原训练样本
plt.scatter(x_fearures[:,0], x_fearures[:,1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
# 上面可视化的决策边界
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')
plt.show()

# predict_proba 函数预测其概率
y_label_new1_predict_proba = lr_clf.predict_proba(x_fearures_new1)
y_label_new2_predict_proba = lr_clf.predict_proba(x_fearures_new2)
print('The New point 1 predict Probability of each class:\n',y_label_new1_predict_proba)
print('The New point 2 predict Probability of each class:\n',y_label_new2_predict_proba)

# 预测类别
y_label_new1_predict = lr_clf.predict(x_fearures_new1)
y_label_new2_predict = lr_clf.predict(x_fearures_new2)
print('The New point 1 predict class:\n',y_label_new1_predict)
print('The New point 2 predict class:\n',y_label_new2_predict)
# 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所有我们可以利用

