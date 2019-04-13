import os 
import tensorflow as tf 
import numpy as np

cwd= 'data/' 
classes={'husky','chihuahua'} #人为 设定 2 类
writer= tf.python_io.TFRecordWriter("zero_train.tfrecord",
                                    options=tf.python_io.TFRecordOptions(
                                        tf.python_io.TFRecordCompressionType.ZLIB)) #要生成的文件

#for index,name in enumerate(classes):
for index in range(1000000):
    example = tf.train.Example(features=tf.train.Features(feature={
        "prices": tf.train.Feature(float_list=tf.train.FloatList(value=[i*0.01 for i in range(100)])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        #'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    })) #example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  #序列化为字符串

writer.close()