import os 
import tensorflow as tf 
#from PIL import Image  #注意Image,后面会用到
#import matplotlib.pyplot as plt 
import numpy as np



def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                                           'mg_raw' : tf.FixedLenFeature([2], tf.float32, default_value=[0.0,0.0]),
                                           #'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    #img = tf.decode_raw(features['img_raw'], tf.uint8)
    #img = tf.reshape(img, [128, 128, 3])  #reshape为128*128的3通道图片
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    img = tf.cast(features['mg_raw'], tf.float32) #在流中抛出label张量
    return img, label

img, label = read_and_decode("dog_train.tfrecords") #要生成的文件
print( "label=" + str(label) )
print( "img=" + str(img) )
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    img2, label2 = sess.run([img,label])#在会话中取出image和label
    print( "label2=" + str(label2) )
    print( "img2=" + str(img2) )
    coord.request_stop()
    coord.join(threads)