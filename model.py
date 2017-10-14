
import tensorflow as tf
import numpy as np

########## CNN ##########

# Convolutional Layer 1
filterSize1 = 5
numFilters1 = 32
stride1_x = 1
stride1_y = 1

# Convolutional Layer 2
filterSize2 = 5
numFilters2 = 64
stride2_x = 1
stride2_y = 1

# Convolutional Layer 3
filterSize3 = 5
numFilters3 = 128
stride3_x = 2
stride3_y = 2


# Convolutional Layer 4
filterSize4 = 3
numFilters4 = 256
stride4_x = 2
stride4_y = 2

# Convolutional Layer 5
filterSize5 = 3
numFilters5 = 512
stride5_x = 2
stride5_y = 2

#Image Dimentions
img_w = 640
img_h = 400
img_size_flat = img_h*img_w
num_channels = 3



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

x = tf.placeholder(tf. float32, shape=[None, img_size_flat], name='x')


x_image = tf.reshape(x, [-1, img_h, img_w, num_channels])

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

layer_conv5, weights_conv5 =     new_conv_layer(input=layer_conv4,
                   num_input_channels=numFilters4,
                   filter_size=filterSize5,
                   num_filters=numFilters5,
                   stride_x=stride5_x,
                   stride_y=stride5_y)

print(layer_conv5)

#transpose_cnn_output = tf.transpose(layer_conv5)
#print(transpose_cnn_output)

conv_shape = tf.shape(layer_conv5)
conv_layer_flat = tf.reshape(layer_conv5, (-1, conv_shape[2]*conv_shape[1], 512))
print(conv_layer_flat)
