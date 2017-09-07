import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg') # Changing matplotlib backend
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:/Projects/s_lstm_gan/data/", one_hot=True)

class s_lstm_gan_mnist_model(object):
	def __init__(self, config, is_training = True, model_type="FULL"):
		batch_size = config.batch_size
		z_size = config.z_size

		lstm_layers_RNN_g = config.lstm_layers_RNN_g
		lstm_layers_RNN_d = config.lstm_layers_RNN_d

		hidden_size_RNN_g = config.hidden_size_RNN_g
		hidden_size_RNN_d = config.hidden_size_RNN_d


		self.target = tf.placeholder(tf.float32, [batch_size, 10])
		self.target_bin = tf.placeholder(tf.float32, [batch_size, 2])
		self.trainables_variables = []

		# Generator_LSTM --
		if model_type == "GEN" or model_type == "FULL":
			self.z = tf.placeholder(tf.float32, [batch_size, z_size])

			# Linear Transformation for Z to hidden size rnn
			f_w = tf.get_variable("RNN_g_w", [z_size, hidden_size_RNN_g])
			f_b = tf.get_variable("RNN_g_b", [hidden_size_RNN_g])

			self.trainables_variables.append(f_w)
			self.trainables_variables.append(f_b)

			init_state = tf.matmul(self.z, f_w) + f_b
			collected_state = ((init_state, init_state),)
			for layer in range(config.lstm_layers_RNN_g - 1):
				collected_state += ((init_state, init_state),)

			init_image = tf.zeros([batch_size,14*14])

			init_input = tf.concat(1, [init_image, self.target])

			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_RNN_g, forget_bias=0.0, state_is_tuple=True)
			cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.lstm_layers_RNN_g, state_is_tuple=True)

			# Linear Transformation for [x_image_size * y_image_size + num_classes] to  hidden size rnn
			g_w = tf.get_variable("RNN_g_input_target_w", [(14*14)+10, hidden_size_RNN_g])
			g_b = tf.get_variable("RNN_g_input_target_b", [hidden_size_RNN_g])

			self.trainables_variables.append(g_w)
			self.trainables_variables.append(g_b)

			# Linear Transformation for hidden size rnn to [x_image_size * y_image_size]
			h_w = tf.get_variable("RNN_g_output_target_w", [hidden_size_RNN_g, (14*14)])
			h_b = tf.get_variable("RNN_g_output_target_b", [(14*14)])

			self.trainables_variables.append(h_w)
			self.trainables_variables.append(h_b)

			output = []
			cell_input = tf.matmul(init_input, g_w) + g_b
			self.state = state = collected_state

			lstm_variables = []

			with tf.variable_scope("RNN_g") as vs:
				for time_step in range(4):
					if time_step > 0: tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(tf.nn.relu(cell_input), state)
					cell_output = tf.matmul(cell_output, h_w) + h_b
					output.append(cell_output)
					new_input = tf.concat(1, [cell_output, self.target])
					cell_input = tf.matmul(new_input, g_w) + g_b

				lstm_variables = [v for v in tf.global_variables()
                    if v.name.startswith(vs.name)]

			self.trainables_variables += lstm_variables

			outputs_RNN_g = tf.transpose(output, perm=[1,0,2])
			outputs_RNN_g = tf.nn.relu(outputs_RNN_g)

			output_max = tf.reduce_max(outputs_RNN_g, reduction_indices=2)
			output_max = tf.expand_dims(output_max, -1)
			output_max = tf.tile(output_max, [1,1,14*14])

			stabalizer = tf.ones(tf.shape(output_max)) * 1e-7

			outputs_RNN_g = tf.div(outputs_RNN_g, output_max + stabalizer)

			if model_type == "GEN":	
				self.outputs = outputs_RNN_g

		

		# 
