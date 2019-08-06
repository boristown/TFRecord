import os 
import tensorflow as tf 
import numpy as np
import csv
import glob
import random
import time

path = 'I:/Training/*/*.csv'
dir = glob.glob(path)
package_size = 10000000
current_size = 0
package_index =1
test_ratio = 0.005 #test data 0.5%
validation_ratio = 0.00 #validation data 0%
train_ratio = 1.0 - test_ratio - validation_ratio #train data 99.5%
random_list = np.random.rand(len(dir))
price_tr_count = 120

tfwriters = {}
tfwritercount = {}
last_train = None
last_test = None

train_count = 0
validation_count = 0
for file_index in range(len(dir)):
    print("processing %s" % dir[file_index])
    random_float = random_list[file_index]
    file_path = dir[file_index]
    prefix = ""
    if(random_float < train_ratio):
        local_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))[0:11]
        prefix="train-"
    elif(random_float < train_ratio + validation_ratio):
        local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))[0:9]
        prefix="dev-"
    else:
        local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        prefix="validation-"

    writer_key = prefix + ("%s" % local_time ) +".tfrecord"
    if writer_key not in tfwriters:
        if prefix == "train-" and last_train != None:
            last_train.close()
        if prefix == "validation-" and last_test != None:
            last_test.close()
        tfwriter = tf.python_io.TFRecordWriter(writer_key,
                                        options=tf.python_io.TFRecordOptions(
                                            tf.python_io.TFRecordCompressionType.ZLIB))
        tfwriters[writer_key] = tfwriter
        tfwritercount[writer_key] = 0
        if prefix == "train-":
            last_train = tfwriter
        if prefix == "validation-":
            last_test = tfwriter

    with open(file_path, 'r') as f:
        csvreader = csv.reader(f)
        csvlist = list(csvreader)
        random.shuffle(csvlist)
        for csvline in csvlist:
            example = tf.train.Example(features=tf.train.Features(feature={
                "prices": tf.train.Feature(float_list=tf.train.FloatList(value=list(map(float,csvline[0:price_tr_count])))),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=list(map(int,csvline[price_tr_count:price_tr_count+1])))),
            }))
            tfwriters[writer_key].write(example.SerializeToString()) 
            tfwritercount[writer_key] += 1
            if prefix == "train-":
                train_count +=1
            else:
                validation_count += 1

last_train.close()
last_test.close()
for writerkey, tfwriter in tfwriters.items():
    print("%s=%d" % (writerkey, tfwritercount[writerkey]))
print("train=%d\nvalidation=%d" % (train_count, validation_count))
