class BiLSTM(object):

	def __init__(self,  num_of_layer, n_hidden, classification_num, n_input):

		self.num_of_layer = num_of_layer
		self.n_hidden = n_hidden
		self.classification_num = classification_num
		self.n_input = n_input


	def __graph__(self):
		self.x = tf.placeholder("float", [None, self.n_input, 1])
		self.y = tf.placeholder("float", [None, self.classification_number])

		self.weights = {
			'out': tf.Variable(tf.random_normal([2*n_hidden, classification_number]))
		}

		self.biases = {
			'out': tf.Variable(tf.random_normal([classification_number]))
		}

		self.pred = Bi_LSTM(x, weights, biases, layer=num_of_layer)

		# Loss and optimizer
		self.softmax_result = tf.nn.softmax(logits=pred)
		self.cost_in_one_batch = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		# Model evaluation
		self.correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Initializing the variables
		init = tf.global_variables_initializer()

		

	def Bi_LSTM(self, x, weights, biases, layer):

	    # x.shape = [10, 200, 1](n_input, batch_size, number_of_tokens in single time step)
	    x = tf.unstack(x, n_input, 1)

	    # Forward direction cell
	    lstm_fw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for ly in range(layer)])
	    # Backward direction cell
	    lstm_bw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for ly in range(layer)])

	    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

	    # there are n_input outputs but we only want the last output
	    return tf.matmul(outputs[-1], weights['out']) + biases['out']
