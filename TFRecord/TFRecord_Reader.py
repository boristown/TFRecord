import os 
import tensorflow as tf 
import numpy as np



def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'prices' : tf.FixedLenFeature([5], tf.float32, default_value=[0.0]*5),
                                           'label': tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                                           #'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    prices = tf.cast(features['prices'], tf.float32) #在流中抛出label张量
    return prices, label

prices, label = read_and_decode("zero_train.tfrecords") #要生成的文件
print( "label=" + str(label) )
print( "prices=" + str(prices) )
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(2):
        prices2, label2 = sess.run([prices,label])#在会话中取出image和label
        print( "label=" + str(label2) )
        print( "prices=" + str(prices2) )
    coord.request_stop()
    coord.join(threads)