import tensorflow as tf 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import Mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:/Reasearch/GAN/Code", one_hot=True)

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
   
    # Generator 
    
    
  
  

