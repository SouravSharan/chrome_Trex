
import tensorflow as tf
import numpy as np
import time
from loaddata_2 import read_images
########## CNN ##########

# Convolutional Layer 1
filterSize1 = 5
numFilters1 = 16
stride1_x = 1
stride1_y = 1

# Convolutional Layer 2
filterSize2 = 5
numFilters2 = 16
stride2_x = 2
stride2_y = 2

# Convolutional Layer 3
filterSize3 = 5
numFilters3 = 32
stride3_x = 2
stride3_y = 2

# Convolutional Layer 4
filterSize4 = 3
numFilters4 = 64
stride4_x = 2
stride4_y = 2

#FC 1
fc_size = 128

#Image Dimentions
img_w = 64
img_h = 64
img_size_flat = img_h*img_w
num_channels = 3

num_classes = 2

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))



def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))



def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,      # Number of filters.
                   stride_x,
                   stride_y):

    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weight (filters)
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, stride_y, stride_x, 1],
                         padding='SAME')

    # A bias-value is added to each filter-channel.
    layer += biases

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.relu(layer)
    return layer

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_h, img_w, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filterSize1,
                   num_filters=numFilters1,
                   stride_x=stride1_x,
                   stride_y=stride1_y  )

print(layer_conv1)

layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=numFilters1,
                   filter_size=filterSize2,
                   num_filters=numFilters2,
                   stride_x=stride2_x,
                   stride_y=stride2_y  )

print(layer_conv2)

layer_conv3, weights_conv3 =     new_conv_layer(input=layer_conv2,
                   num_input_channels=numFilters2,
                   filter_size=filterSize3,
                   num_filters=numFilters3,
                   stride_x=stride3_x,
                   stride_y=stride3_y  )

print(layer_conv3)


layer_conv4, weights_conv4 =     new_conv_layer(input=layer_conv3,
                   num_input_channels=numFilters3,
                   filter_size=filterSize4,
                   num_filters=numFilters4,
                   stride_x=stride4_x,
                   stride_y=stride4_y  )

print(layer_conv4)

conv_shape = tf.shape(layer_conv4)
layer_flat, num_features = flatten_layer(layer_conv4)

print(layer_flat, num_features)


layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size)

print(layer_fc1)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes)

print(layer_fc2)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 48
total_iterations = 0

saver = tf.train.Saver()

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch = read_images('folder', train_batch_size)
        images, labels = session.run([x_batch, y_true_batch])
        feed_dict_train = {x: images, y: labels}
        print(feed_dict_train)
        input()
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
            saver.save(session, 'tf-model')

    total_iterations += num_iterations

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(10)
