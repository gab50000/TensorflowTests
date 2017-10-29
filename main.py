import tensorflow as tf
import numpy as np


def neuro():
    n_in = 4
    n_hidden = 3
    n_out = 2
    initializer = tf.contrib.layers.variance_scaling_initializer()
    input_ = tf.placeholder(dtype=tf.float32, shape=[None, n_in])
    hidden = tf.layers.dense(input_, n_hidden, activation=tf.nn.sigmoid)
    output = tf.layers.dense(hidden, 2)
    probs = tf.nn.softmax(output)
    return input_, output


def optimizer():
    learning_rate = 0.01
    return tf.train.AdamOptimizer(learning_rate)


def main():
    input_, net = neuro()    
    opt = optimizer()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        test_val = np.array([1, 2, 3, 4], dtype=float).reshape((-1, 4))
        probs = sess.run(net, feed_dict={input_: test_val})
        #  print(opt.compute_gradients(net))
        grads = [sess.run(grad, feed_dict={input_: test_val}) for grad, var in opt.compute_gradients(net)]

    print(f"Probs = {probs}")
    print(f"Grads: {grads}")


if __name__ == "__main__":
    main()
