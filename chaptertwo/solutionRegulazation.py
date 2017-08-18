
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

import matplotlib.pyplot as plt

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


  image_size = 28
  num_labels = 10

  def reformat(dataset, labels):
      dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
      # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
      labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
      return dataset, labels


  train_dataset, train_labels = reformat(train_dataset, train_labels)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
  test_dataset, test_labels = reformat(test_dataset, test_labels)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


  def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])


#   batch_size = 128
#
#   graph = tf.Graph()
#   with graph.as_default():
#       # Input data. For the training data, we use a placeholder that will be fed
#       # at run time with a training minibatch.
#       tf_train_dataset = tf.placeholder(tf.float32,
#                                         shape=(batch_size, image_size * image_size))
#       tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#       tf_valid_dataset = tf.constant(valid_dataset)
#       tf_test_dataset = tf.constant(test_dataset)
#       beta_regul = tf.placeholder(tf.float32)
#
#       # Variables.
#       weights = tf.Variable(
#           tf.truncated_normal([image_size * image_size, num_labels]))
#       biases = tf.Variable(tf.zeros([num_labels]))
#
#       # Training computation.
#       logits = tf.matmul(tf_train_dataset, weights) + biases
#       ##this one using l2 regularzation
#       loss = tf.reduce_mean(
#           tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + beta_regul * tf.nn.l2_loss(weights)
#
#       # Optimizer.
#       optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#       # Predictions for the training, validation, and test data.
#       train_prediction = tf.nn.softmax(logits)
#       valid_prediction = tf.nn.softmax(
#           tf.matmul(tf_valid_dataset, weights) + biases)
#       test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
#
#   num_steps = 3001
#
#   with tf.Session(graph=graph) as session:
#       tf.initialize_all_variables().run()
#       print("Initialized")
#       for step in range(num_steps):
#           # Pick an offset within the training data, which has been randomized.
#           # Note: we could use better randomization across epochs.
#           offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#           # Generate a minibatch.
#           batch_data = train_dataset[offset:(offset + batch_size), :]
#           batch_labels = train_labels[offset:(offset + batch_size), :]
#           # Prepare a dictionary telling the session where to feed the minibatch.
#           # The key of the dictionary is the placeholder node of the graph to be fed,
#           # and the value is the numpy array to feed to it.
#           feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regul: 1e-3}
#           _, l, predictions = session.run(
#               [optimizer, loss, train_prediction], feed_dict=feed_dict)
#           if (step % 500 == 0):
#               print("Minibatch loss at step %d: %f" % (step, l))
#               print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#               print("Validation accuracy: %.1f%%" % accuracy(
#                   valid_prediction.eval(), valid_labels))
#       print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
#
#   num_steps = 3001
#   regul_val = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
#   accuracy_val = []
#
# # 对于同一批data，取不同的 regular value，实验出最好的效果
#   for regul in regul_val:
#       with tf.Session(graph=graph) as session:
#           tf.initialize_all_variables().run()
#           for step in range(num_steps):
#               # Pick an offset within the training data, which has been randomized.
#               # Note: we could use better randomization across epochs.
#               offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#               # Generate a minibatch.
#               batch_data = train_dataset[offset:(offset + batch_size), :]
#               batch_labels = train_labels[offset:(offset + batch_size), :]
#               # Prepare a dictionary telling the session where to feed the minibatch.
#               # The key of the dictionary is the placeholder node of the graph to be fed,
#               # and the value is the numpy array to feed to it.
#               feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regul: regul}
#               _, l, predictions = session.run(
#                   [optimizer, loss, train_prediction], feed_dict=feed_dict)
#           accuracy_val.append(accuracy(test_prediction.eval(), test_labels))
#           print("accuray is %.1f%% with regul value is %.5f%%" % (accuracy(test_prediction.eval(), test_labels) , regul))

  # batch_size = 128
  # num_hidden_nodes = 1024
  #
  # graph = tf.Graph()
  # with graph.as_default():
  #     # Input data. For the training data, we use a placeholder that will be fed
  #     # at run time with a training minibatch.
  #     tf_train_dataset = tf.placeholder(tf.float32,
  #                                       shape=(batch_size, image_size * image_size))
  #     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  #     tf_valid_dataset = tf.constant(valid_dataset)
  #     tf_test_dataset = tf.constant(test_dataset)
  #     beta_regul = tf.placeholder(tf.float32)
  #
  #     # Variables.
  #     weights1 = tf.Variable(
  #         tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
  #     biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
  #     weights2 = tf.Variable(
  #         tf.truncated_normal([num_hidden_nodes, num_labels]))
  #     biases2 = tf.Variable(tf.zeros([num_labels]))
  #
  #     # Training computation.
  #     lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  #     logits = tf.matmul(lay1_train, weights2) + biases2
  #     loss = tf.reduce_mean(
  #         tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + \
  #            beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
  #
  #     # Optimizer.
  #     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  #
  #     # Predictions for the training, validation, and test data.
  #     train_prediction = tf.nn.softmax(logits)
  #     lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  #     valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
  #     lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  #     test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)
  #
  #     num_steps = 3001
  #
  #     with tf.Session(graph=graph) as session:
  #         tf.initialize_all_variables().run()
  #         print("Initialized")
  #         for step in range(num_steps):
  #             # Pick an offset within the training data, which has been randomized.
  #             # Note: we could use better randomization across epochs.
  #             offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
  #             # Generate a minibatch.
  #             batch_data = train_dataset[offset:(offset + batch_size), :]
  #             batch_labels = train_labels[offset:(offset + batch_size), :]
  #             # Prepare a dictionary telling the session where to feed the minibatch.
  #             # The key of the dictionary is the placeholder node of the graph to be fed,
  #             # and the value is the numpy array to feed to it.
  #             feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regul: 1e-3}
  #             _, l, predictions = session.run(
  #                 [optimizer, loss, train_prediction], feed_dict=feed_dict)
  #             if (step % 500 == 0):
  #                 print("Minibatch loss at step %d: %f" % (step, l))
  #                 print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
  #                 print("Validation accuracy: %.1f%%" % accuracy(
  #                     valid_prediction.eval(), valid_labels))
  #         print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  # batch_size = 128
  # num_hidden_nodes = 1024
  #
  # graph = tf.Graph()
  # with graph.as_default():
  #     # Input data. For the training data, we use a placeholder that will be fed
  #     # at run time with a training minibatch.
  #     tf_train_dataset = tf.placeholder(tf.float32,
  #                                       shape=(batch_size, image_size * image_size))
  #     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  #     tf_valid_dataset = tf.constant(valid_dataset)
  #     tf_test_dataset = tf.constant(test_dataset)
  #     beta_regul = tf.placeholder(tf.float32)
  #
  #     # Variables.
  #     weights1 = tf.Variable(
  #         tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
  #     biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
  #     weights2 = tf.Variable(
  #         tf.truncated_normal([num_hidden_nodes, num_labels]))
  #     biases2 = tf.Variable(tf.zeros([num_labels]))
  #
  #     # Training computation.
  #     lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  #     ##adding dropout
  #     drop1 = tf.nn.dropout(lay1_train,0.5)
  #     logits = tf.matmul(drop1, weights2) + biases2
  #     loss = tf.reduce_mean(
  #         tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf_train_labels))
  #
  #     # Optimizer.
  #     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  #
  #     # Predictions for the training, validation, and test data.
  #     train_prediction = tf.nn.softmax(logits)
  #     lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  #     valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
  #     lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  #     test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)
  #
  #
  #   # when the batch size become small,
  #   # but we have drop out
  # num_steps = 101
  # num_batches = 3
  #
  # with tf.Session(graph=graph) as session:
  #     tf.initialize_all_variables().run()
  #     print("Initialized")
  #     for step in range(num_steps):
  #         # Pick an offset within the training data, which has been randomized.
  #         # Note: we could use better randomization across epochs.
  #         # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
  #         offset = ((step % num_batches) * batch_size) % (train_labels.shape[0] - batch_size)
  #         # Generate a minibatch.
  #         batch_data = train_dataset[offset:(offset + batch_size), :]
  #         batch_labels = train_labels[offset:(offset + batch_size), :]
  #         # Prepare a dictionary telling the session where to feed the minibatch.
  #         # The key of the dictionary is the placeholder node of the graph to be fed,
  #         # and the value is the numpy array to feed to it.
  #         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regul: 1e-3}
  #         _, l, predictions = session.run(
  #             [optimizer, loss, train_prediction], feed_dict=feed_dict)
  #         if (step % 2 == 0):
  #             print("Minibatch loss at step %d: %f" % (step, l))
  #             print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
  #             print("Validation accuracy: %.1f%%" % accuracy(
  #                 valid_prediction.eval(), valid_labels))
  #     print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  ## multiply layer with dropouts and learning rate decay
  batch_size = 128
  num_hidden_nodes1 = 1024
  num_hidden_nodes2 = 256
  num_hidden_nodes3 = 128
  keep_prob = 0.5

  graph = tf.Graph()
  with graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      global_step = tf.Variable(0)

      # Variables.
      weights1 = tf.Variable(
          tf.truncated_normal(
              [image_size * image_size, num_hidden_nodes1],
              stddev=np.sqrt(2.0 / (image_size * image_size)))
      )
      biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
      weights2 = tf.Variable(
          tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
      biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
      weights3 = tf.Variable(
          tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
      biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]))
      weights4 = tf.Variable(
          tf.truncated_normal([num_hidden_nodes3, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes3)))
      biases4 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
      lay2_train = tf.nn.relu(tf.matmul(lay1_train, weights2) + biases2)
      lay3_train = tf.nn.relu(tf.matmul(lay2_train, weights3) + biases3)
      logits = tf.matmul(lay3_train, weights4) + biases4
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

      # Optimizer.
      learning_rate = tf.train.exponential_decay(0.5, global_step, 4000, 0.65, staircase=True)
      # use learning rate decay
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
      lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, weights2) + biases2)
      lay3_valid = tf.nn.relu(tf.matmul(lay2_valid, weights3) + biases3)
      valid_prediction = tf.nn.softmax(tf.matmul(lay3_valid, weights4) + biases4)
      lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
      lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights2) + biases2)
      lay3_test = tf.nn.relu(tf.matmul(lay2_test, weights3) + biases3)
      test_prediction = tf.nn.softmax(tf.matmul(lay3_test, weights4) + biases4)

  num_steps = 18001

  with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
          # Pick an offset within the training data, which has been randomized.
          # Note: we could use better randomization across epochs.
          offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
          # Generate a minibatch.
          batch_data = train_dataset[offset:(offset + batch_size), :]
          batch_labels = train_labels[offset:(offset + batch_size), :]
          # Prepare a dictionary telling the session where to feed the minibatch.
          # The key of the dictionary is the placeholder node of the graph to be fed,
          # and the value is the numpy array to feed to it.
          feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
          _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
          if (step % 500 == 0):
              print("Minibatch loss at step %d: %f" % (step, l))
              print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
              print("Validation accuracy: %.1f%%" % accuracy(
                  valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))