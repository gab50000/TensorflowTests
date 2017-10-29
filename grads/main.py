import tensorflow as tf
import numpy as np


def main():
    input_ = tf.placeholder(dtype=tf.float32, shape=[1])
    W = tf.get_variable("W", [1])
    out = tf.nn.sigmoid(W * input_)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        x = sess.run(out, feed_dict={input_: np.array([5])})
        print(f"Output: {x}")
        print(f"W = {W.eval()}")


if __name__ == "__main__":
    main()
