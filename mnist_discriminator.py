import tensorflow as tf
import numpy as np
from our_ops import *
import tflib as lib

slim = tf.contrib.slim

class MNIST_Discriminator():
	
	# def discriminator(x, labels, batch_size, var_coll, fc_space_size):
	#   result = slim.layers.convolution(
	#           x,
	#           32,
	#           kernel_size=[5, 5],
	#           stride=1,
	#           padding='SAME',
	#           activation_fn=tf.nn.relu,
	#           weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
	#           variables_collections=var_coll
	#       )

	#   result = slim.layers.max_pool2d(result, [2, 2])

	#   result = slim.layers.convolution(
	#           x,
	#           64,
	#           kernel_size=[5, 5],
	#           stride=1,
	#           padding='SAME',
	#           activation_fn=tf.nn.relu,
	#           weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
	#           variables_collections=var_coll
	#       )

	#   result = tf.reshape(result, [batch_size, -1])
	#   result = slim.layers.fully_connected(result, fc_space_size, activation_fn=tf.nn.relu,
	#       variables_collections=var_coll)
	#   labels = slim.fully_connected(labels, fc_space_size, activation_fn=tf.nn.relu,
	#       variables_collections=var_coll)
	#   result = tf.concat([result, labels], 1)
	#   result = slim.layers.fully_connected(result, 2*fc_space_size,
	#       activation_fn=tf.nn.relu, variables_collections=var_coll)
	#   result = slim.layers.fully_connected(result, 1, activation_fn=None,
	#       variables_collections=var_coll)
	#   return result
	def __init__(self):
		pass

	def discriminator(self, x, labels, labels_size, batch_size, fc_space_size, n_layers=3):
		print(x.shape)
		OUTPUT_DIM = 28*28*1
		FC_DIM = 512
		x = tf.reshape(x, [-1, OUTPUT_DIM])
		output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, x)
		for i in xrange(n_layers):
			output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
		output = lib.ops.linear.Linear('Discriminator.PreOut', FC_DIM, fc_space_size, output)
		labels = lib.ops.linear.Linear('Discriminator.Labels', 11, fc_space_size, labels)
		output = tf.concat([output, labels], axis=1)
		output = lib.ops.linear.Linear('Discriminator.Out', 2*fc_space_size, 1, output)
		return tf.reshape(output, [-1])