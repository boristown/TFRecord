import os 
import tensorflow as tf 
import numpy as np

def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader(
                                    options=tf.python_io.TFRecordOptions(
                                        tf.python_io.TFRecordCompressionType.ZLIB))
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           #'prices' : tf.FixedLenFeature([100], tf.float32, default_value=[0.0]*100),
                                           'prices' : tf.VarLenFeature(tf.float32),
                                           'label': tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                                       })#将image数据和label取出来
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    price_dense = tf.sparse.to_dense(features['prices']) #在流中抛出label张量
    return price_dense, label

filepath = "G:\Robot\TFRecord\TFRecord\validation-2020010622.tfrecord"
prices, label = read_and_decode(filepath) #要生成的文件
print( "label=" + str(label) )
print( "prices=" + str(prices) )

with tf.Session() as sess: #开始一个会话
    #init_op = tf.initialize_all_variables()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(100):
        #print (sess.run(label))
        #print (sess.run(prices))
        try:
            label2 = sess.run(label)#在会话中取出image和label
            prices2 = sess.run([prices])#在会话中取出image和label
        except:
            print ("Error %d" % i);
        else:
            print( "label=" + str(label2) )
            #print( "labels=" + str(labels3) )
            print( "prices=" + str(prices2) )
        
    coord.request_stop()
    coord.join(threads)