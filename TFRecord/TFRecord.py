import os 
import tensorflow as tf 
import numpy as np
import csv
import glob
import random
import time

starttime = time.time()
path = 'G:/Training/*/*.csv'
dir = glob.glob(path)
package_size = 10000000
current_size = 0
package_index =1
test_ratio = 0.00 #0.02 #test data 2%
validation_ratio = 0.00 #validation data 0%
train_ratio = 1.0 - test_ratio - validation_ratio #train data 99.5%
random_list = np.random.rand(len(dir))
price_tr_count = 120
max_price_count = 10000
tfwriters = {}
tfwritercount = {}
last_train = None
last_test = None

train_count = 0
validation_count = 0
for file_index in range(len(dir)):
    percent = (file_index+1)*100.0/len(dir)
    currenttime = time.time()
    runningtime =  currenttime - starttime
    lefttime = runningtime * (100 - percent) / percent
    runningtimestr = time.strftime('%H:%M:%S', time.gmtime(runningtime))
    lefttimestr = time.strftime('%H:%M:%S', time.gmtime(lefttime))
    print("processing %d of %d[%f%%]Time=%s;Left=%s" % (file_index+1 , len(dir), percent, runningtimestr, lefttimestr))
    random_float = random_list[file_index]
    file_path = dir[file_index]
    prefix = ""
    if "validation-" in file_path:
        local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        prefix="validation-"
    else:
        #local_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))[0:11]
        local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        prefix="train-"

    '''
    if(random_float < train_ratio):
        local_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))[0:11]
        prefix="train-"
    elif(random_float < train_ratio + validation_ratio):
        local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))[0:9]
        prefix="dev-"
    else:
        local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        prefix="validation-"
    '''

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
        #csvreader = csv.reader(f)
        #csvlist = list(csvreader)
        pricelist = f.read().splitlines()
        
        #random.shuffle(csvlist)
        days = len(pricelist)

        if days > max_price_count:
            pricelist = pricelist[-max_price_count: : 1]
            days = max_price_count
        if days < max_price_count:
            pricelist = pricelist + [0]*(max_price_count - days)

        feature = [float(i) for i in pricelist]
        label = days
        
        '''
        frame_feature = list(map(lambda id: tf.train.Feature(float_list=tf.train.FloatList(value=[id])), feature))
        
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}),
            feature_lists=tf.train.FeatureLists(feature_list={
                'prices': tf.train.FeatureList(feature=frame_feature)
            })
        )

        '''

        #for closeprice in pricelist:
        
        example = tf.train.Example(features=tf.train.Features(feature={
            "prices": tf.train.Feature(float_list=tf.train.FloatList(value=list(map(float,pricelist)))),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=list(map(int,[label])))),
        }))

        tfwriters[writer_key].write(example.SerializeToString()) 
        tfwritercount[writer_key] += 1
        

        if prefix == "train-":
            train_count += 1
        else:
            validation_count += 1

if last_train:
    last_train.close()
if last_test:
    last_test.close()
for writerkey, tfwriter in tfwriters.items():
    print("%s=%d" % (writerkey, tfwritercount[writerkey]))
print("train=%d\nvalidation=%d" % (train_count, validation_count))
