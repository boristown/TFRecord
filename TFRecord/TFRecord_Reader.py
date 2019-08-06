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
                                           'prices' : tf.FixedLenFeature([100], tf.float32, default_value=[0.0]*100),
                                           'label': tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                                           #'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    prices = tf.cast(features['prices'], tf.float32) #在流中抛出label张量
    return prices, label

prices, label = read_and_decode("I:\\TFRECORDS\\Processed\\train-20190524073.tfrecord") #要生成的文件
print( "label=" + str(label) )
print( "prices=" + str(prices) )
operations = tf.reshape(label, [1,1])
#operations = tf.tile(operations, [MAX_CASE, LABEL_COUNT])
#op_list = np.zeros([MAX_CASE, LABEL_COUNT])
operationMod = tf.mod(operations, 2)
operationDiv = tf.floordiv(operations, 2)
labels = tf.identity(operationMod)
for i in range(10-1):
   operationMod = tf.mod(operationDiv, 2)
   operationDiv = tf.floordiv(operationDiv, 2)
   #op_list[i][1] = parsed['label']
   labels = tf.concat([operationMod, labels], 0)
   tf.logging.info("labels=%s" % (labels.shape))
   #for j in range(MAX_CASE-1-i):
   #    operations[i][1] = tf.floordiv(operations[i][1], 2)
   #operations[i][1] = tf.mod(operations[i][1], 2)
   #operations[i][0] = tf.subtract(1, op_list[i][1])
#operations = tf.cast(tf.stack(op_list), tf.int32)
labels2 = tf.subtract(1, labels)
labels = tf.concat([labels2, labels], 1)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(100):
        try:
            prices2, label2, labels3 = sess.run([prices,label,labels])#在会话中取出image和label
        except:
            print ("Error %d" % i);
        else:
            print( "label=" + str(label2) )
            print( "labels=" + str(labels3) )
            print( "prices=" + str(prices2) )
    coord.request_stop()
    coord.join(threads)