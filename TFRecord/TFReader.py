import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = 'G:\Robot\TFRecord\TFRecord\validation-2020010622.tfrecord'  # address to save the hdf5 file

with tf.Session() as sess:
    #feature = {'train/image': tf.FixedLenFeature([], tf.string),
    #           'train/label': tf.FixedLenFeature([], tf.int64)}
    feature ={
                        'prices' : tf.VarLenFeature(tf.float32),
                        'label': tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                    }
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    #image = tf.decode_raw(features['train/image'], tf.float32)
    
    # Cast label data into int32
    # label = tf.cast(features['train/label'], tf.int32)
    prices = tf.cast(features['prices'], tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # Reshape image data into the original shape
    #image = tf.reshape(image, [224, 224, 3])
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    pricesTensor, labels = tf.train.shuffle_batch([prices, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(5):
        pri, lbl = sess.run([pricesTensor, labels])

        #pri = pri.astype(np.uint8)

        for j in range(6):
            #plt.subplot(2, 3, j+1)
            #plt.imshow(pri[j, ...])
            #plt.title('cat' if lbl[j]==0 else 'dog')
            print(lbl,pri)

        #plt.show()

    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()
