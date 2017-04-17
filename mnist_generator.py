import tensorflow as tf
import numpy as np
import tflib as lib
from our_ops import *

slim = tf.contrib.slim

class MNIST_Generator():

	def __init__(self):
		pass

	def generator(self, z, noise_dim):
		FC_DIM = 512
		output = ReLULayer('Generator.1', noise_dim, FC_DIM, z)
		output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
		output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
		output = lib.ops.linear.Linear('Generator.Out', FC_DIM, 28*28*1, output)
		output = tf.tanh(output)
		output = tf.reshape(output, [-1, 28, 28, 1])
		return output