import os
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model(object):
    def __init__(self, train_batcher=None, test_batcher=None):
        self.batcher = train_batcher
        self.test_batcher = test_batcher
        self.window_size = 5
        self.number_of_features = 464

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.window_size, self.number_of_features])
            self.y = tf.placeholder(tf.float32, [None, 2])

            x_flatted = tf.reshape(self.x, [-1, self.window_size * self.number_of_features])

            w = {
                "1": tf.Variable(tf.random_normal([self.window_size * self.number_of_features, self.window_size * self.number_of_features])),
                "2": tf.Variable(tf.random_normal([self.window_size * self.number_of_features, self.window_size * self.number_of_features])),
                "3": tf.Variable(tf.random_normal([self.window_size * self.number_of_features, 2])),
            }

            b = {
                "1": tf.Variable(tf.random_normal([self.window_size * self.number_of_features])),
                "2": tf.Variable(tf.random_normal([self.window_size * self.number_of_features])),
                "3": tf.Variable(tf.random_normal([2]))
            }

            y1 = tf.nn.tanh(tf.matmul(x_flatted, w["1"]) + b["1"])
            y2 = tf.nn.tanh(tf.matmul(y1, w["2"]) + b["2"])
            y_ = tf.matmul(y2, w["3"]) + b["3"]

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_))
            self.train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.predict = tf.argmax(tf.nn.softmax(y_), 1)

            predictions = tf.argmax(y_, 1)
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

                print(sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y}))
                print(sess.run([self.tp, self.tn, self.fp, self.fn], feed_dict={self.x: batch_x, self.y: batch_y}))

                saver = tf.train.Saver()
                saver.save(sess, "model/epoch", global_step=e)

    def test(self, x):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "data/epoch-37")

            return sess.run(self.predict, feed_dict={self.x: x})
