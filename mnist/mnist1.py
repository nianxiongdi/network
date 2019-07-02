# -*- coding: utf-8 -*-#

"""
File         :      mnist1.py
Description  :  
Author       :      赵金朋
Modify Time  :      2019/7/2 11:21
"""
"""
1.定义算法公式，也就是神经网络的forward时的计算
2.定义loss，选定优化器，并制定优化器优化loss
3.迭代对数据进行训练
4.在测试集或者验证集对准确率进行评测
"""
#加载数据集
from tensorflow.examples.tutorials.mnist import  input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#输出大小
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

"""
softmax
"""
import  tensorflow as tf
sess=tf.InteractiveSession()
#placeholder微数据输入的地方，第一个参数为数据类型，第二个为数据的尺寸，none为不限条件输入，输入为784维向量
x=tf.placeholder(tf.float32,[None,784])


#权重偏置初始化为0
w=tf.Variable(tf.zeros([784,10]))#特征数784*维数10
b=tf.Variable(tf.zeros([10]))
#y=softmax(w*x+b)
#tf.matmul矩阵乘法
y=tf.nn.softmax(tf.matmul(x,w)+b)

#cross_entropy作为多分类的损失函数
y_=tf.placeholder(tf.float32,[None,10])
#求均值，
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
#梯度下降算法，学习率0.5，优化目标cross_entropy
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#全局参数初始化
tf.global_variables_initializer().run()
#迭代训练train_step，每次抽取100个构成mini-batch，选取少部分收敛速度快，容易跳出局部最小
for i in  range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
#tf.argmax(y,1)预测概率最大的，tf.argmax(y_,1)真实类别
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))