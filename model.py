import tensorflow as tf
from tensorflow.contrib import rnn

class Model(object):
    def __init__(self, train_batcher=None, test_batcher=None):
        self.batcher = train_batcher
        self.test_batcher = test_batcher
        self.number_of_timesteps = 8
        self.number_of_features = 550

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.number_of_timesteps, self.number_of_features])
            self.y = tf.placeholder(tf.float32, [None, 2])

            number_of_hidden = 512
            w = { "out": tf.Variable(tf.random_normal([2 * number_of_hidden, 2])) }
            b = { "out": tf.Variable(tf.random_normal([2])) }

            def BiRNN(x, weights, biases):
                # Prepare data shape to match `rnn` function requirements
                # Current data input shape: (batch_size, timesteps, n_input)
                # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

                # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
                x = tf.unstack(x, self.number_of_timesteps, 1)

                # Define lstm cells with tensorflow
                # Forward direction cell
                lstm_fw_cell = rnn.BasicLSTMCell(number_of_hidden, forget_bias=1.0)
                # Backward direction cell
                lstm_bw_cell = rnn.BasicLSTMCell(number_of_hidden, forget_bias=1.0)

                # Get lstm cell output
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

                # Linear activation, using rnn inner loop last output
                return tf.matmul(outputs[-1], weights['out']) + biases['out']

            y_ = BiRNN(self.x, w, b)  # logits
            prediction = tf.nn.softmax(y_)

            # Define loss and optimizer
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))
            optimizer = tf.train.AdamOptimizer()
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_step = optimizer.minimize(loss_op)

            # Evaluate model (with test y_, for dropout to be disabled)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            self.predict = tf.argmax(prediction, 1)

            predictions = tf.argmax(prediction, 1)
            actuals = tf.argmax(self.y, 1)

            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predictions = tf.ones_like(predictions)
            zeros_like_predictions = tf.zeros_like(predictions)

            self.tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                           tf.equal(predictions, ones_like_predictions)), "float"))

            self.tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                           tf.equal(predictions, zeros_like_predictions)), "float"))

            self.fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                           tf.equal(predictions, ones_like_predictions)), "float"))

            self.fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                           tf.equal(predictions, zeros_like_predictions)), "float"))

    def train(self):
        with tf.Session(graph=self.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for e in range(100):
                print("Epoch {e}".format(e=e))

                print("# of batches {n}".format(n=self.batcher.get_number_of_batches()))

                self.batcher.reset_batches()
                while self.batcher.has_more_batches():
                    batch_x, batch_y = self.batcher.get_next_batch()
                    sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})

                # Test trained model
                self.test_batcher.reset_batches()
                batch_x, batch_y = self.test_batcher.get_all()

                print(sess.run([self.accuracy, self.tp, self.tn, self.fp, self.fn],
                               feed_dict={self.x: batch_x, self.y: batch_y}))

                saver = tf.train.Saver()
                saver.save(sess, "data/model/epoch", global_step=e)

    def test(self, x):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "data/model/epoch-37")

            return sess.run(self.predict, feed_dict={self.x: x})
