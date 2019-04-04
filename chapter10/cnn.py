import os
import tensorflow as tf
from functools import reduce
from tensorflow.contrib.layers.python.layers import initializers


class CNN(Network):
  def __init__(self, sess,
               data_format,
               history_length,
               observation_dims,
               output_size, 
               trainable=True,
               hidden_activation_fn=tf.nn.relu,
               output_activation_fn=None,
               weights_initializer=initializers.xavier_initializer(),
               biases_initializer=tf.constant_initializer(0.1),
               value_hidden_sizes=[512],
               advantage_hidden_sizes=[512],
               network_output_type='dueling',
               network_header_type='nips',
               name='CNN'):
    super(CNN, self).__init__(sess, name)

    if data_format == 'NHWC':
      self.inputs = tf.placeholder('float32',
          [None] + observation_dims + [history_length], name='inputs')
    elif data_format == 'NCHW':
      self.inputs = tf.placeholder('float32',
          [None, history_length] + observation_dims, name='inputs')
    else:
      raise ValueError("unknown data_format : %s" % data_format)

    self.var = {}
    self.l0 = tf.div(self.inputs, 255.)

    with tf.variable_scope(name):
      if network_header_type.lower() == 'nature':
        self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
            32, [8, 8], [4, 4], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l1_conv')
        self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
            64, [4, 4], [2, 2], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l2_conv')
        self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2,
            64, [3, 3], [1, 1], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l3_conv')
        self.l4, self.var['l4_w'], self.var['l4_b'] = \
            linear(self.l3, 512, weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l4_conv')
        layer = self.l4
      elif network_header_type.lower() == 'nips':
        self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
            16, [8, 8], [4, 4], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l1_conv')
        self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
            32, [4, 4], [2, 2], weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l2_conv')
        self.l3, self.var['l3_w'], self.var['l3_b'] = \
            linear(self.l2, 256, weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l3_conv')
        layer = self.l3
      else:
        raise ValueError('Wrong DQN type: %s' % network_header_type)

      self.build_output_ops(layer, network_output_type,
          value_hidden_sizes, advantage_hidden_sizes, output_size,
          weights_initializer, biases_initializer, hidden_activation_fn,
          output_activation_fn, trainable)




class Network(object):
  def __init__(self, sess, name):
    self.sess = sess
    self.copy_op = None
    self.name = name
    self.var = {}

  def build_output_ops(self, input_layer, network_output_type, 
      value_hidden_sizes, advantage_hidden_sizes, output_size, 
      weights_initializer, biases_initializer, hidden_activation_fn, 
      output_activation_fn, trainable):
    
    self.outputs, self.var['w_out'], self.var['b_out'] = linear(input_layer, output_size, weights_initializer,
                 biases_initializer, output_activation_fn, trainable, name='out')
  

    self.max_outputs = tf.reduce_max(self.outputs, reduction_indices=1)
    self.outputs_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
    self.outputs_with_idx = tf.gather_nd(self.outputs, self.outputs_idx)
    self.actions = tf.argmax(self.outputs, axis=1)

  def run_copy(self):
    if self.copy_op is None:
      raise Exception("run `create_copy_op` first before copy")
    else:
      self.sess.run(self.copy_op)

  def create_copy_op(self, network):
    with tf.variable_scope(self.name):
      copy_ops = []

      for name in self.var.keys():
        copy_op = self.var[name].assign(network.var[name])
        copy_ops.append(copy_op)

      self.copy_op = tf.group(*copy_ops, name='copy_op')

  def calc_actions(self, observation):
    return self.actions.eval({self.inputs: observation}, session=self.sess)

  def calc_outputs(self, observation):
    return self.outputs.eval({self.inputs: observation}, session=self.sess)

  def calc_max_outputs(self, observation):
    return self.max_outputs.eval({self.inputs: observation}, session=self.sess)

  def calc_outputs_with_idx(self, observation, idx):
    return self.outputs_with_idx.eval(
        {self.inputs: observation, self.outputs_idx: idx}, session=self.sess)



def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d',
           trainable=True):
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, 
        tf.float32, initializer=weights_initializer, trainable=trainable)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

    b = tf.get_variable('b', [output_dim],
        tf.float32, initializer=biases_initializer, trainable=trainable)
    out = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b

def linear(input_,
           output_size,
           weights_initializer=initializers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=None,
           trainable=True,
           name='linear'):
  shape = input_.get_shape().as_list()

  if len(shape) > 2:
    input_ = tf.reshape(input_, [-1, reduce(lambda x, y: x * y, shape[1:])])
    shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('w', [shape[1], output_size], tf.float32,
        initializer=weights_initializer, trainable=trainable)
    b = tf.get_variable('b', [output_size],
        initializer=biases_initializer, trainable=trainable)
    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b

def batch_sample(probs, name='batch_sample'):
  with tf.variable_scope(name):
    uniform = tf.random_uniform(tf.shape(probs), minval=0, maxval=1)
    samples = tf.argmax(probs - uniform, dimension=1)
  return samples
