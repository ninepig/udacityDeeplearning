import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    #softmax 会让所有输出加起来成为1，方便概率
    #softmax 全部参数乘以10 会让结果更加偏向0或者1
    #softmax 全部参数除以10 会让结果更加偏向uniform
    #softmax 函数是一个指数函数, 也就是当x越来越大的时候 分子增长的幅度越快
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x = np.arange(-2.0, 6.0, 0.1)
# print(x)
# print(np.ones_like(x))
# print(0.2 * np.ones_like(x))
# vstack 就是把输入竖向变成新的数组，这里变成了[[1],[2],[3]]
# 竖向的数组 丢给softmax 根据矩阵进行计算，也就是针对 y1 y2 y3 （x代表的）同时算3个值，算出来以后成为了softmax的矩阵，但这个时候是60个元素每个矩阵
#为了显示好看 我们进行T转置，这样变成了[[X1,X2,X3],[X1,X2,X3]]这样的形式，方便输出
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
# print(scores)
# print(softmax(scores))
# print(softmax(scores).T)
plt.plot(x,softmax(scores).T,linewidth=2)
# plt.plot(x,softmax(scores),linewidth=2)
plt.show()
