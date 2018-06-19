import tensorflow as tf
class Foo(object):
    def __init__(self, *args, **kwargs):
        self.qqq = tf.Variable(
            tf.zeros((10,)),
            name='qqq', trainable=False)
        self.aaa = 0
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    @classmethod
    def work(cls, aaa):
        with tf.Session() as sess:
            print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            # sess.run(self.qqq)
            # print(self.aaa)
