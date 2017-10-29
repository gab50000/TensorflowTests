import tensorflow as tf
import numpy as np
import daiquiri


daiquiri.setup(level=daiquiri.logging.INFO)
logger = daiquiri.getLogger(__name__)


def Neuro():
    n_in = 4
    n_hidden = 3
    n_out = 2
    initializer = tf.contrib.layers.variance_scaling_initializer()
    input_ = tf.placeholder(dtype=tf.float32, shape=[None, n_in], name="Input")
    hidden = tf.layers.dense(input_, n_hidden, activation=tf.nn.sigmoid, kernel_initializer=initializer,
                             name="Hidden")
    output = tf.layers.dense(hidden, 2, kernel_initializer=initializer, name="Output")
    probs = tf.nn.softmax(output, name="Softmax")
    return input_, probs


def Optimizer():
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate)


def get_gradients(opt, net):
    gv = opt.compute_gradients(out)
    grad_placeholders = []
    for grad, var in gv:
        placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        grad_placeholders.append(placeholder)


def main():
    logs_path = "."
    input_, out = Neuro()

    optimizer = Optimizer()
    gv1 = optimizer.compute_gradients(out[0, 0])
    gv2 = optimizer.compute_gradients(out[0, 1])
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    arr_in = np.array([1, 2, 3, 4], dtype=float)[None, :]
    logger.info(sess.run(out, feed_dict={input_: arr_in}))
    updated_grads = [(-1 * g.eval(feed_dict={input_: arr_in}), var) for g, var in gv1]
    training_op = optimizer.apply_gradients(updated_grads)
    sess.run(training_op)
    logger.info(sess.run(out, feed_dict={input_: arr_in}))

    logger.debug([gr[0].eval(feed_dict={input_: arr_in}) for gr in gv1])
    logger.debug([gr[0].eval(feed_dict={input_: arr_in}) for gr in gv2])
    sess.close()

if __name__ == "__main__":
    main()
