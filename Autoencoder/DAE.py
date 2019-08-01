# -*- coding: utf-8 -*-#

"""
File         :      DAE1.py
Description  :  TF4.2自编码器
Author       :      赵金朋
Modify Time  :      2019/7/12 10:23
"""
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#xavier初始化器
def xavier_init(fan_in, fan_out, constant=1):
    # np.sqrt 平方根运算  fan_in  输入节点的数量   fan_out输出节点的数量
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    #均匀分布random_uniform
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


'''
这里的自编码器采用Xavier initialization方法初始化参数，需要先定义好它。
Xavier初始化器的特点是会根据某一层网络的输入、输出节点数量自动整合最合适的分布。使得初始化的权重不大不小，正好合适。
从数学的角度分析，Xavier就是让权重满足0均值，同时方差为 2/(n[input] + n[output]),其中n为节点数，分布可以用均匀分布或高斯分布。

'''

#下面定义一个去噪自编码器的类，这个类包含构建函数init()和一些常用的成员函数。
class AdditiveGaussianNoiseAutoencoder(object):
    #构建函数init()
    '''
    init函数包含这样几个输入：
    n_input（输入变量数）
    n_hidden（隐含层节点数）
    transfer_function（隐含层激活函数，默认为softplus）
    optimizer（优化器，默认为Adam）
    scale（高斯噪音系数，默认为0.1）。
    其中，class中的scale参数做成了一个placeholder，参数初始化使用接下来定义的_initialize_weights函数。
    只使用一个隐含层。
    '''
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),
                 scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()#初始化权重的函数_initialize_weights
        self.weights = network_weights

        '''
        接下里开始定义网络结构，我们为输入x创建一个维度为n_input的placeholder。
        然后建立一个能提取特征的隐含层，先将x加上噪音，即 self.x+scale*tf.random_normal((n_input,)),
        然后用tf.mutmul将加了噪音的输入和隐含层的权重相乘，加上bias，最后使用transfer进行激活处理。
        经过隐含层后，我们要在输出层进行数据复原，重建操作（建立reconstruction层），这里不需要激活函数，直接将隐含层的s输出self.hidden乘以输出层的权重w2，加上bias2
        '''
        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input, )),self.weights['w1']),
                self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])


        '''
        定义自编码器的损失函数，直接使用平方误差作为cost
        计算输出self.reconstruction和输入self.x之间的差(tf.subtract求差)，再用tf.pow求平方，
        最后用tf.reduce_sum求和即可得到平方误差。定义self.optimizer作为优化器对self.cost进行优化。
        '''
        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        '''
        最后创建Session，初始化自编码器全部模型参数
        '''
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    '''
    初始化权重的函数_initialize_weights:
    先创建一个all_weights的字典，把w1，w2，b1,b2全部放进去，
    最后返回all_weights其中w1需要使用前面提到的xavier__init函数初始化，
    直接传入输入层节点数和隐含层节点数然后Xavier即可返回一个比较适合softplus等激活函数的权重初始分布，
    偏置b1直接用tf.zeros全部初始化为0。
    输出层self.reconstruction因为没有使用激活函数，直接w2和b2初始化为0即可。
    '''
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights


    '''
    定义计算损失cost以及执行一步训练的函数partial_fit。
    函数里只需要让Session执行两个计算图的节点损失cost和训练过程optimizer
    输入的feed_dict字典包括输出数据x和噪音系数scale。
    函数partial_fit做的就是用一个batch数据进行训练并返回当前的损失cost
    '''
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X,self.scale: self.training_scale})
        return cost
    '''
    只求损失cost的函数calc_total_cost
    只让Session执行一个计算图节点self.cost，传入的参数和前面的partial_fit一样。
    这个函数是自编码器训练完成后，在测试集上对模型性能进行评测的时候会用到的，不会像partial_fit触发训练操作
    '''
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,self.scale: self.training_scale})

    '''
    定义transform函数，返回自编码器隐含层的输出结果。
    目的是提供一个借口来获取抽象后的特征，自编码器的隐含层的最主要功能就是学习数据中的高阶特征
    '''
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,self.scale: self.training_scale})


    '''
    定义generate函数。它将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据。
    这个接口和前面的transform刚好将整个自编码器拆分为两部分，这里的generate为后部分，把高阶特征复原为原始数据的部分
    '''
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    '''定义reconstruct函数，它整体运行一遍复原过程'''
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X,self.scale: self.training_scale})
    '''获取隐含层的权重w1的函数'''
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    '''获取隐含层权重b1的函数'''
    def getBiases(self):
        return self.sess.run(self.weights['b1'])





'''
定义一个对训练，测试数据进行标准化处理的函数。
标准化即让数据变成0均值，且标准差为1的分布.方法就是先减去均值，再除以标准差。
直接使用sklearn.preprocessing的StandardScaler这个类现在训练集上进行fit，z再将这个Scaler用到训练数据和测试数据上。
这里要注意的是，必须要保证训练集和测试集都使用相同的Scaler，这样才能保证后面处理数据的一致性。
这也是为什么先在训练数据上fit出一个共用的Scaler的原因
'''
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test
'''
定义一个随机获取block数据的函数：
取一个0到batch_size之间的随机整数，再以这个整数作为block的起始位置，然后顺序得到一个batch_size数据。
这属于不放回抽样，提高数据的利用效率和随机性
'''
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


if __name__ == '__main__':
    # 加载数据集
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    # 对数据进行标准化变换
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    # 定义几个常用参数：总训练样本数，最大训练轮数（epoch），batch_size，设置每隔一轮就显示一次损失cost
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1
    # 创建一个自编码器的实例：
    # 定义模型的输入节点n_input为784
    # 自编码器的隐含层节点数n_hidden为200
    # 隐含层激活函数为softplus
    # 优化器为Adam且学习率为0.001
    # 噪音系数scale设为0.01
    autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                                   n_hidden=200,
                                                   transfer_function=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                   scale=0.01)

    for epoch in range(training_epochs):
        avg_cost = 0.  #设置平均损失函数为0
        total_batch = int(n_samples / batch_size)  #计算总共需要的batch数  使用放回抽样 不能保证每个样本都被抽到并参与训练
        # 所有batches的循环
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)  #随机抽取一个block的数据
            cost = autoencoder.partial_fit(batch_xs)#autoencoder的成员函数partial_fit训练这个batch的数据并计算当前的cost
            avg_cost += cost / n_samples * batch_size # 计算平均loss

        # Display logs per epoch step  显示当前迭代数和这一轮迭代的平均cost
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    #性能测试          calc_total_cost对测试集X_test进行测试  评价指数：平方误差
    print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
