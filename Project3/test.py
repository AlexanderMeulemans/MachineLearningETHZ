import tensorflow as tf
import numpy as np


class CNN(object):
    def __init__(self, patch_size, num_filters_first_layer, num_filters_second_layer,
                 size_fully_connected_layer, num_classes=10, image_size=784, lambda_reg=0.001):
        # Placeholders for input of images, labels and dropout rate
        self.x = tf.placeholder(tf.float32, shape=[None, image_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.keep_prob = tf.placeholder(tf.float32, shape={})

        # creates and returns a weight variable with given shape initialized with
        # a truncated normal distribution with stddev of 0.1
        def weight_variable(shape, nameVar):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=nameVar)

        # creates and returns a bias variable with given shape initialized with
        # a constant of 0.1
        def bias_variable(shape, nameVar):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=nameVar)

        # computes a 2D convolution for the input data x and the filter W
        # uses a stride of one and is zero padded so the output is the same size as the input
        # input shapes:
        # x is the input tensor - should be a 4-D tensor of shape [batch_size, in_height, in_width, in_channels]
        # W is the filter tensor - should be a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        # strides is a 4-D tensor that defines how the filter slides over the input tensor in each of the 4 dimensions
        # padding - if it is set to "SAME" it means that zero padding on every side of the input is introduced to
        # make the shapes match if needed such that the filter is centered at all the pixels of the image according
        # to the strides.
        # ex. if strides=[1, 1, 1, 1] and padding='SAME' the filter is centered at every pixel from the image
        # padding - if it is set to "VALID" it means that there is no padding.
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        # performs max pooling over 2x2 blocks
        # input shapes:
        # x is the input tensor - should be a 4-D tensor of shape [batch_size, in_height, in_width, in_channels]
        # ksize has the same dimensionality as the input tensor. It defines the patch size. It extracts the max
        # value out of each such patch. Here the patch we define is a 2x2 block
        # strides is a 4-D tensor that defines how the patch slides over the input tensor
        # if the padding is "SAME" there is padding, if it is "VALID" there is no padding
        # For the SAME padding, the output height and width are computed as:
        #     out_height = ceil(float(in_height) / float(strides1))
        #     out_width = ceil(float(in_width) / float(strides[2]))
        # For the VALID padding, the output height and width are computed as:
        #     out_height = ceil(float(in_height - filter_height + 1) / float(strides1))
        #     out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
        # example: if x is an image of shape [2,3] and has 1 channel (so the input shape is [1, 2, 3, 1])
        # , we max pool with 2x2 kernel and the stride is 2
        # if the pad is VALID (valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID'))
        #  the output is of shape [1, 1, 1, 1]
        # if the pad is SAME we pad the image to the shape [2, 4];
        # (same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')) the output is of shape [1, 1, 2, 1]
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Create a convolution + maxpool layer for the first layer

        # define the filter
        self.W_conv1 = weight_variable([patch_size, patch_size, 1, num_filters_first_layer], "filter_layer1")
        b_conv1 = bias_variable([num_filters_first_layer], "bias_layer1")
        # reshape the data to a 4D tensor to fit into the convolution
        # the second and third dimensions correspond to image width and height,
        #  and the final dimension corresponds to the number of color channels
        # the first dimension is for the batch size; when we have -1 for one dimension when reshaping
        # it will dynamically calculate that dimension
        # example: if x is of shape [a, b*c, d] and we run tf.reshape([-1, b, c, d]), the first dimension will be "a"
        # this is useful when the batch size varies
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        # apply convolution, add the bias, apply relu, and then max pooling
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + b_conv1)
        # print h_conv1.get_shape() # the shape is [-1, 28, 28, 32]
        h_pool1 = max_pool_2x2(h_conv1)
        # print h_pool1.get_shape() # the shape is [-1, 14, 14, 32]

        # Create a second layer of convolution + maxpool
        # define the filter
        self.W_conv2 = weight_variable(
            [patch_size, patch_size, num_filters_first_layer, num_filters_second_layer], "filter_layer2")
        b_conv2 = bias_variable([num_filters_second_layer], "bias_layer2")
        # apply convolution, add the bias, apply relu, and then max pooling
        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + b_conv2)
        # print h_conv2.get_shape() # the shape is [-1,  14, 14, num_filters_first_layer] i.e. [-1,  14, 14, 64]
        h_pool2 = max_pool_2x2(h_conv2)
        # print h_pool2.get_shape() # the shape is [-1,  7, 7, num_filters_second_layer] i.e. [-1, 7, 7, 64]

        # Create a densely connected layer
        W_fc1 = weight_variable([7 * 7 * 64, size_fully_connected_layer], "W_fc1")
        b_fc1 = bias_variable([size_fully_connected_layer], "b_fc1")

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # the shape of h_fc1 is [-1, size_fully_connected_layer]

        # Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([size_fully_connected_layer, num_classes], "W_fc2")
        b_fc2 = bias_variable([num_classes], "b_fc2")

        self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # Regularizer
        self.l2_loss = tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2)

        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1])) + lambda_reg * self.l2_loss
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime
import time
import sys
import os

# ==================== Parameters =====================================

# Model Hyperparameters
tf.flags.DEFINE_integer("num_filters_first_layer", 32, "Number of filters per filter size for first layer(default: 32)")
tf.flags.DEFINE_integer("num_filters_second_layer", 64, "Number of filters per filter size for 2nd layer (default: 64)")
tf.flags.DEFINE_integer("patch_size", 5, "Size of the filter (default: 5)")
tf.flags.DEFINE_integer("size_fully_connected_layer", 512, "Size of the fully connected layer (default: 1024)")
tf.flags.DEFINE_float("keep_prob", 0.8, "dropout strength on fully connected layer")
tf.flags.DEFINE_float("lambda_reg", 0.001, "Regularization strength for L2 regularizer")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 2000)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("run_name", None, "Suffix for output directory. If None, a timestamp is used instead")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# ==================== Data Loading and Preparation ===================

# downloads the MNIST data if it doesn't exist
# each image is of size 28x28, and is stored in a flattened version (size 784)
# the label for each image is a one-hot vector (size 10)
# the data is divided in training set (mnist.train) of size 55,000, validation set
# (mnist.validation) of size 5,000 and test set (mnist.test) of size 10,000
# for each set the images and labels are given (e.g. mnist.train.images of size
# [55,000, 784] and mnist.train.labels of size [55,000, 10])
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    # Create a session
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Build the graph
        cnn = CNN(patch_size=FLAGS.patch_size, num_filters_first_layer=FLAGS.num_filters_first_layer,
                  num_filters_second_layer=FLAGS.num_filters_second_layer,
                  size_fully_connected_layer=FLAGS.size_fully_connected_layer,
                  lambda_reg=FLAGS.lambda_reg)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.cross_entropy)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        if FLAGS.run_name is None:
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        else:
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.run_name))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.cross_entropy)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        l2_summary = tf.summary.scalar("l2_regularizer", cnn.l2_loss)

        # Step time
        step_time_placeholder = tf.placeholder(dtype=tf.float32, shape={})
        step_time_summary = tf.summary.scalar("step_time", step_time_placeholder)
        last_step_time = 0.0

        # Since we have many channels, we will get filters only for one channel
        V = tf.slice(cnn.W_conv1, (0, 0, 0, 0), (-1, -1, -1, 1))
        # Bring into shape expected by image_summary
        V = tf.reshape(V, (-1, 5, 5, 1))
        image_summary_op = tf.summary.image("kernel_layer1", V, 5)
        # TODO: Add a summary for visualizing the filters from the second layer
        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, image_summary_op, grad_summaries_merged, step_time_summary, l2_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. TF assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()

        # Initialize all the variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
                A single training step
                """
            feed_dict = {
                cnn.x: x_batch,
                cnn.y_: y_batch,
                step_time_placeholder: last_step_time,
                cnn.keep_prob: FLAGS.keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.cross_entropy, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            train_summary_writer.flush()


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.x: x_batch,
                cnn.y_: y_batch,
                cnn.keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.cross_entropy, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        for i in range(FLAGS.num_epochs):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            begin = time.time()
            train_step(batch[0], batch[1])
            end = time.time()
            last_step_time = end - begin

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(mnist.validation.images, mnist.validation.labels, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        print("test accuracy %g" % cnn.accuracy.eval(feed_dict={
            cnn.x: mnist.test.images, cnn.y_: mnist.test.labels, cnn.keep_prob: 1.0}))