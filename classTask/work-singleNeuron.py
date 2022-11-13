import numpy as np
import os
import matplotlib.pyplot as plt
path = 'data/AbaloneAgePrediction.txt'
#处理数据集
def loadData():
    # 读取文件
    file = open(path, 'r')
    dataFile = file.read()

    # 将性别（M：雄性，F：雌性，I：未成年）映射成数字 0 和 1
    dataFile = dataFile.replace("M", '1')
    dataFile = dataFile.replace("F",'0')
    dataFile = dataFile.replace("I", '0') # 未成年归为雌性
    # dataFile = dataFile.replace("I", '1')# 未成年归为雄性

    dataFile = dataFile.replace("\n", ',')
    dataFile = dataFile.split(',')
    del dataFile[-1] # 最后一个换行被置换为逗号，有作为分割符号使后面多了一个空格

    # 转换为nparray
    data = np.array(dataFile, dtype=float)

    featureNames = ['性别', '长度', '直径', '高度', '总重量', '皮重', '内脏重量', '克重', '年龄']
    featureNum = len(featureNames)

    #  data.reshape(int(data.shape[0]/featureNum), featureNum)# 不用等号赋值就不会对数组产生影响
    data = data.reshape(int(data.shape[0]/featureNum), featureNum)

    # 将训练数据集和测试数据集按照8:2的比例分开
    ratio = 0.8  # 训练集的占比
    offset = int(data.shape[0]*ratio)  # 训练集的行数
    training_data = data[:offset]  # 从零开始到offset结束的部分，不包括第offset
    # 用训练集的数值对训练集进行归一化
    # 数据归一化
    maximums = training_data.max(axis=0)  # 沿垂直方向的最小值
    minimums = training_data.min(axis=0)  # 沿垂直方向的最大值
    avgs = training_data.sum(axis=0) / training_data.shape[0]  # 沿垂直方向的平均值
    for i in range(featureNum):
        data[:, i] = (data[:, i]-avgs[i])/(maximums[i]-minimums[i])  # -0.5到0.5
        # data[:, i] = 2*(data[:, i] - avgs[i]) / (maximums[i] - minimums[i])  # -1到1
        # data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])  # 0到1

    # 分割训练集、测试集
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

# 定义一个单神经元类
class Network(object):
    '''参数数量'''
    # 定义初始化函数
    def __init__(self, num_of_weights):
        self.w = np.random.randn(num_of_weights,1)  #每个参数的w
        self.b = 0
    # 定义向前运算函数
    # 根据 输入的数据，现有的wb 预测结果
    def forward(self,x):
        z = np.dot(x, self.w) + self.b
        return z
    # 定义loss函数
    def loss(self,z,y):
        error = z-y  # 预测值与实际值的茶
        cost = error*error  # 差的平方
        cost = np.mean(cost)  # 平均值
        return cost
    # 定义梯度计算函数
    #x为训练集参数 y为训练集结果
    def gradient(self,x,y):
        z = self.forward(x)  # 用现有wb与训练集参数预测出结果z

        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]

        gradient_b = (z-y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b
# 定义梯度下降法更新参数函数
    def update(self, gradient_w, gradient_b,eta):
        self.w = self.w - eta*gradient_w
        self.b = self.b - eta*gradient_b

# 定义训练函数
    def train(self, x, y, interations, eta):
        losses = []
        for i in range(interations):
            z = self.forward(x)
            L = self.loss(z,y)
            gradient_w, gradient_b = self.gradient(x,y)
            self.update(gradient_w,gradient_b,eta)
            losses.append(L)
            if (i+1)%10 ==0:
                print('iter{},loss{}'.format(i,L))
        return losses
# 主函数
if __name__ == '__main__':
    train_data, test_data = loadData()

    x = train_data[:, :-1]
    y = train_data[:, -1:]
# 定义网络对象
    net = Network(8)
    num_iterations = 1000
# 启动训练
    losses = net.train(x, y, num_iterations, eta= 0.01)
# 把每轮训练的loss值用曲线形式展示出来
    plot_x = np.arange(num_iterations)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()
