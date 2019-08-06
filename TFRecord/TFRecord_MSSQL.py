import os 
import tensorflow as tf 
import numpy as np
import csv
import glob
import random
import time
import pymssql

#conn = pymssql.connect(host='127.0.0.1',user='sa',
#                      password='hello',database='NPKW',
#                      charset="utf8")

package_size = 3
package_count = 2

current_size = 0
package_index =1
test_ratio = 0.005 #test data 0.5%
validation_ratio = 0.00 #validation data 0%
train_ratio = 1.0 - test_ratio - validation_ratio #train data 99.5%
price_tr_count = 120

tfwriters = {}
tfwritercount = {}
last_train = None
last_test = None

train_count = 0
validation_count = 0

serverName = '127.0.0.1'
hostName = r'PC-20170613DZEI\ROBOT'
userName = r'PC-20170613DZEI\Administrator'

#Conn_string = 'data source="PC-20170613DZEI\ROBOT"; initial catalog="Robot"; trusted_connection=True' 

for package_index in range(0, package_count):

    random_list = np.random.rand(package_size)

    conn = pymssql.connect(server=serverName,database="Robot")
    #conn = pymssql.connect(
    #    host=hostName, 
    #    user=userName, 
    #    password='',
    #    database="Robot"
    #    )

    print(conn)

    cursor = conn.cursor()
    sql = 'select top ' + package_size  + ' * from TFRecords ORDER BY NEWID()'
    cursor.execute(sql)
    #用一个rs变量获取数据
    rs = cursor.fetchall()
    print(rs)
    row_index = 0
    while rs:
        #生成tfrecord行
        prefix = ""
        random_float = random_list[row_index]
        if(random_float < train_ratio):
            local_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))[0:11]
            prefix="train-"
        elif(random_float < train_ratio + validation_ratio):
            local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))[0:9]
            prefix="dev-"
        else:
            local_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))
            prefix="validation-"

        writer_key = prefix + ("%s" % local_time ) +".tfrecords"
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
        example = tf.train.Example(features=tf.train.Features(feature={
                "prices": tf.train.Feature(float_list=tf.train.FloatList(value=list(map(float,rs[1:price_tr_count+1])))),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=list(map(int,csvline[price_tr_count+1:price_tr_count+2])))),
            }))
        tfwriters[writer_key].write(example.SerializeToString()) 
        tfwritercount[writer_key] += 1
        if prefix == "train-":
            train_count +=1
        else:
            validation_count += 1

        print("ID=%d, Label=%s" % (rs[0], rs[121]))
        row_index += 1
    conn.close()

last_train.close()
last_test.close()

for writerkey, tfwriter in tfwriters.items():
    print("%s=%d" % (writerkey, tfwritercount[writerkey]))
print("train=%d\nvalidation=%d" % (train_count, validation_count))
