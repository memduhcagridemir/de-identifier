import datetime

import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, train_batcher=None, test_batcher=None):
        self.number_of_hidden = 256
        self.number_of_timesteps = 9
        self.number_of_features = 551
        self.batcher = train_batcher
        self.test_batcher = test_batcher

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32, [None, self.number_of_timesteps, self.number_of_features])

            with tf.name_scope('target'):
                self.y = tf.placeholder(tf.float32, [None, 2])

            with tf.name_scope('weights'):
                w1 = tf.Variable(tf.random_normal([2 * self.number_of_hidden, 2]))

            with tf.name_scope('biases'):
                b1 = tf.Variable(tf.random_normal([2]))

            with tf.name_scope('operations'):
                x = tf.unstack(self.x, self.number_of_timesteps, 1)

                lstm_fw_cell = rnn.BasicLSTMCell(self.number_of_hidden, forget_bias=1.0)
                lstm_bw_cell = rnn.BasicLSTMCell(self.number_of_hidden, forget_bias=1.0)

                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

            with tf.name_scope('output'):
                y_ = tf.matmul(outputs[self.number_of_timesteps // 2], w1) + b1
                probs = tf.nn.softmax(y_)

            with tf.name_scope('loss'):
                loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))
                tf.summary.scalar("loss", loss_op)

            with tf.name_scope('train'):
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                self.train_step = tf.train.AdamOptimizer().minimize(loss_op)

            correct_pred = tf.equal(tf.argmax(probs, 1), tf.argmax(self.y, 1))

            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            self.predict = tf.argmax(probs, 1)

            predictions = tf.argmax(probs, 1)
            actuals = tf.argmax(self.y, 1)

            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predictions = tf.ones_like(predictions)
            zeros_like_predictions = tf.zeros_like(predictions)

            with tf.name_scope('scores'):
                self.tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                               tf.equal(predictions, ones_like_predictions)), "float"))
                tf.summary.scalar("tp", self.tp)
                self.tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                               tf.equal(predictions, zeros_like_predictions)), "float"))
                tf.summary.scalar("tn", self.tn)
                self.fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),
                                                               tf.equal(predictions, ones_like_predictions)), "float"))
                tf.summary.scalar("fp", self.fp)
                self.fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                                               tf.equal(predictions, zeros_like_predictions)), "float"))
                tf.summary.scalar("fn", self.fn)

            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/{time}-{number_of_hidden}-{number_of_timesteps}".format(
                time=datetime.datetime.now(),
                number_of_hidden=self.number_of_hidden, number_of_timesteps=self.number_of_timesteps),
                graph=self.graph)

    def train(self):
        with tf.Session(graph=self.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for e in range(100):
                print("Epoch {e}".format(e=e))
                print("# of batches {n}".format(n=self.batcher.get_number_of_batches()))

                self.batcher.reset_batches()
                losses = []
                while self.batcher.has_more_batches():
                    batch_x, batch_y = self.batcher.get_next_batch()
                    sum, batch_loss = sess.run([self.summary_op, self.train_step], feed_dict={self.x: batch_x, self.y: batch_y})
                    losses.append(batch_loss)

                # Test trained model
                self.test_batcher.reset_batches()
                batch_x, batch_y = self.test_batcher.get_all()

                sum, acc, tp, tn, fp, fn = sess.run([self.summary_op, self.accuracy, self.tp, self.tn, self.fp,
                                                     self.fn], feed_dict={self.x: batch_x, self.y: batch_y})
                self.writer.add_summary(sum, global_step=e)
                print(acc, tp, tn, fp, fn)

                saver = tf.train.Saver()
                saver.save(sess, "data/model/epoch", global_step=e)

    def test(self, x):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "data/model/epoch-37")

            return sess.run(self.predict, feed_dict={self.x: x})
