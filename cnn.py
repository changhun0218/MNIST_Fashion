"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from datetime import datetime
now = datetime.utcnow().strftime("%Y%M%D%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run.{}/".format(root_logdir, now)

tf.set_random_seed(2)
np.random.seed(2)


#mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)

"""
train = np.array(pd.read_csv("fashion-mnist_train.csv"))
train_x = train[:, 1:]
train_y0 = train[:, 0]
train_y = np.zeros((train_x.shape[0], 10))
train_y[np.arange(train_x.shape[0]), train_y0] = 1

test = np.array(pd.read_csv("fashion-mnist_test.csv"))
test_x = test[:, 1:]
test_y0 = test[:, 0]
test_y = np.zeros((test_x.shape[0], 10))
test_y[np.arange(test_x.shape[0]), test_y0] = 1
"""

def read_features_from_csv(filename,usecols = range(1, 785)):
    features = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)
    features = np.divide(features, 255.0)
    return features

def read_labels_from_csv(filename):
    labels_original = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=0, dtype=np.int)
    labels = np.zeros([len(labels_original),10])
    labels[np.arange(len(labels_original)), labels_original] = 1
    labels = labels.astype(np.float32)
    return labels

features = read_features_from_csv("fashion-mnist_train.csv")
labels = read_labels_from_csv('fashion-mnist_train.csv')

#input to the graph
x = tf.placeholder(tf.float32, shape = [None,784])
y_ = tf.placeholder(tf.float32, shape = [None,10])

#reshape the  x to feaature 2d image
x_image = tf.reshape(x, [-1, 28,28, 1])

## plot one example
#plt.imshow(mnist_train_x[0].reshape((28, 28)), cmap='gray')
#plt.title('%i' % np.argmax(mnist_train_y[0])); plt.show()

#convolutional layer 1
w_conv1 = tf.Variable( tf.truncated_normal( [5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable( tf.constant(0.1, shape=[32]))

h_conv1 = tf.nn.relu( tf.nn.conv2d(input = x_image, filter =w_conv1, strides=[ 1, 1, 1, 1], padding="SAME") + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#convolutional layer 2
w_conv2 = tf.Variable( tf.truncated_normal( [5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable( tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu( tf.nn.conv2d(input = h_pool1, filter = w_conv2, strides=[ 1, 1, 1, 1], padding="SAME") + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding="SAME")


#fully connected layer 1
w_fc1 = tf.Variable( tf.truncated_normal([7 * 7 * 64, 1024],stddev = 0.1))
b_fc1 = tf.Variable( tf.constant(0.1, shape = [1024]))

h_pool2_flat =  tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu( tf.matmul( h_pool2_flat, w_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#fully connected layer2
w_fc2 = tf.Variable( tf.truncated_normal( [1024, 10], stddev = 0.1))
b_fc2 = tf.Variable( tf.constant( 0.1, shape = [10]))

y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

cross_entropy  = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=y, labels= y_))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction  = tf.equal( tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#hyperparameters which place main role in deciding the correctness / accuracy of your model
BatchSize = 50
TrainSplit = 0.999
TrainigStep = 300

def generate_batch(features, labels, batch_size):
    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]
    return (batch_features, batch_labels)
#split the data into training and validation
train_samples = int( len(features) / (1 / TrainSplit))

train_features = features[: train_samples]
train_labels   = labels[: train_samples]

validation_features = features[train_samples: ]
validation_labels = labels[train_samples: ]
accuracy_history = []
for i in range(TrainigStep):    
    batch_features, batch_labels = generate_batch(train_features, train_labels, BatchSize)
    
    if  i % 50 == 0:
        accuracy_ = sess.run(accuracy, feed_dict = {x : validation_features, y_: validation_labels, keep_prob:1.0})
        accuracy_history.append(accuracy_)
        save_path = saver.save(sess, "/tmp/my_model.ckpt")
        #summary_str = sess.run(accuracy, feed_dict = {x : validation_features, y_: validation_labels, keep_prob:1.0})
        #file_writer.add_summary(summary_str, i)
        print("step  %i  and validation acc :%g "%(i, accuracy_))
    save_path = saver.save(sess, "tmp/my_model_final.ckpt")

    sess.run(train_step, feed_dict = { x: batch_features, y_: batch_labels, keep_prob:0.5})

test_x = read_features_from_csv('fashion-mnist_test.csv')
test_y = read_labels_from_csv('fashion-mnist_test.csv')
#accuracy for test data
acc = accuracy.eval(feed_dict={x:test_x, y_:test_y, keep_prob:1.0})

print("acc:", acc)
