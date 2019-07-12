# -*- coding: utf-8 -*-#

"""
File         :      DAE.py
Description  :  4.2实现自编码器
Author       :      赵金朋
Modify Time  :      2019/7/10 8:31
"""
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xvaier_init(fan_in,fan_out,constant=1):
    """
    实现标准的均匀分布Xaiver初始化器
    """
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    """
    去噪自编码
    """
    #只有一个隐藏层
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.trainning_scale=scale
        network_weights=self._initialize_weights()
        self.weights=network_weights
        #网络结构
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matmul(
            self.x+scale*tf.random_normal((n_input,)),
            self.weights['w1']),self.weights['b']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,
                                self.weights['w2']),self.weights['b2'])
        #损失函数
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(
                self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    #初始化权重
    def _initialize_weight(self):
        all_weights=dict()
        all_weights['w1']=tf.Variable(xvaier_init(self.n_input,self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],
                                               dtype=tf.float32))
        return all_weights

    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),
                    feed_dict={self.x:X,self.scale:self.trainning_scale})
        return cost

    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,
                            self.scale:self.trainning_scale})

    #返回输出
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,
                                self.scale:self.trainning_scale})
    #将隐藏层的输出结果作为输入
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,
                        feed_dict={self.x:X,self.scale:self.trainning_scale})
    #获取隐藏层权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    #获取隐藏层偏执系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    mnist=input_data.read_data_sets('"../MNIST_data/',one_hot=True)

    def standard_scale(X_train,X_test):
        preprocessor=prep.StandardScaler().fit(X_train)
        X_train=preprocessor.transform(X_train)
        X_test=preprocessor.transform(X_test)
        return X_train,X_test
    def get_random_block_from_data(data,batch_size):
        start_index=np.random.randint(0,len(data)-batch_size)
        return data[start_index:(start_index+batch_size)]

    X_train,X_test=standard_scale(mnist.train.image,mnist.test.images)

    n_samples=int(mnist.train.num_examples)