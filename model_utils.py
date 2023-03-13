# encoding = utf8
import tensorflow as tf


def highway(input1, input2, size):
    ss = tf.shape(input1)
    input1 = tf.reshape(input1, shape=[ss[0]*ss[1], size])
    # input2 = Dense(input2, size)
    input2 = tf.reshape(input2, shape=[ss[0]*ss[1], size])

    W = tf.Variable(tf.random_normal([size, size], stddev=0.1))
    U = tf.Variable(tf.random_normal([size, size], stddev=0.1))
    V = tf.Variable(tf.random_normal([size, 1], stddev=0.1))
    t = tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(input1, W) + tf.matmul(input2, U)), V))
    output = input1 * t + input2 * (1 - t)
    output = tf.reshape(output, shape=[ss[0], ss[1], size])

    return output


def Dense(inputs, ouput_size, bias=True):
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, \
                         tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                        )
    return outputs


def self_attention(inputs, size):
    inputs_shape = tf.shape(inputs)
    Q = Dense(inputs, size)
    Q = tf.reshape(Q, (-1, inputs_shape[0] * inputs_shape[1], inputs_shape[2]))
    K = Dense(inputs, size)
    K = tf.reshape(K, (-1, inputs_shape[0] * inputs_shape[1], inputs_shape[2]))
    V = Dense(inputs, size)
    V = tf.reshape(V, (-1, inputs_shape[0] * inputs_shape[1], inputs_shape[2]))
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size))
    A = tf.nn.softmax(A)
    O = tf.matmul(A, V)
    O = tf.reshape(O, (-1, inputs_shape[1], size))
    return O
