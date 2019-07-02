# -*- coding: utf-8 -*-#

"""
File         :      test.py
Description  :  
Author       :      赵金朋
Modify Time  :      2019/7/2 8:30
"""
import tensorflow as tf
hello = tf.constant('hello,tensorf')
sess = tf.Session()
print(sess.run(hello))