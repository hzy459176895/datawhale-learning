

"""
机器学习之决策树实现案例
"""

# 基础函数库
import numpy as np
# 导入画图库
import matplotlib.pyplot as plt
# 导入决策树模型函数
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 构造数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 1, 0, 1, 0, 1])

# 数据样本散点图
plt.figure()
plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
plt.show()

# 调用逻辑回归模型
tree_clf = DecisionTreeClassifier()
# 用决策树拟合构造的数据集
tree_clf = tree_clf.fit(x_fearures, y_label)

# 可视化决策树
# 1 pip install graphviz
# 2 下载：https://www2.graphviz.org/Packages/stable/windows/10/cmake/Release/x64/graphviz-install-2.44.1-win64.exe 安装
# 3 电脑环境变量PATH添加安装位置的bin位置：比如：D:\UserProgramFile\graphviz\bin\
import graphviz
import matplotlib.pyplot as plt
import pydotplus
model = tree_clf
dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf('D:/Pycharm Project/datawhale-learning/ml/tree_rule.pdf')

from sklearn.tree import export_graphviz
with open("D:/Pycharm Project/datawhale-learning/ml/tree_rule.dot", "w") as wFile:
   export_graphviz(tree_clf,out_file=wFile)

## 创建新样本
x_fearures_new1 = np.array([[0, -1]])
x_fearures_new2 = np.array([[2, 1]])
## 在训练集和测试集上分布利用训练好的模型进行预测
y_label_new1_predict = tree_clf.predict(x_fearures_new1)
y_label_new2_predict = tree_clf.predict(x_fearures_new2)
print('The New point 1 predict class:\n',y_label_new1_predict)
print('The New point 2 predict class:\n',y_label_new2_predict)