import os 
import tensorflow as tf 
import numpy as np
import csv
import glob
import random

path = 'I:/Training/20190414/*.csv' 
dir = glob.glob(path)
package_size = 10000000
current_size = 0
package_index =0
test_ratio = 0.01 #test data 1%
validation_ratio = 0.01 #validation data 1%
train_ratio = 1.0 - test_ratio - validation_ratio #train data 98%
random_list = np.random.rand(len(dir))

tfwriters = {}
tfwritercount = {}

for file_index in range(len(dir)):
    print("processing %s" % dir[file_index])
    random_float = random_list[file_index]
    file_path = dir[file_index]
    prefix = ""
    if(random_float < train_ratio):
        prefix="train-"
    elif(random_float < train_ratio + eval_ratio):
        prefix="dev-"
    else:
        prefix="validation-"
    writer_key = prefix + ("%03d" % package_index) +".tfrecord"
    if writer_key not in tfwriters:
        tfwriter = tf.python_io.TFRecordWriter(writer_key,
                                        options=tf.python_io.TFRecordOptions(
                                            tf.python_io.TFRecordCompressionType.ZLIB))
        tfwriters[writer_key] = tfwriter
        tfwritercount[writer_key] = 0

    with open(file_path, 'r') as f:
        csvreader = csv.reader(f)
        csvlist = list(csvreader)
        for csvline in csvlist:
            example = tf.train.Example(features=tf.train.Features(feature={
                "prices": tf.train.Feature(float_list=tf.train.FloatList(value=[csvline[0:100]])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[csvline[100:101]])),
        }))
        tfwriters[writer_key].write(example.SerializeToString()) 

for writerkey, tfwriter in tfwriters.iteritems():
    print("%s=%d" % writerkey, tfwritercount[writerkey])
    tfwriter.close()
