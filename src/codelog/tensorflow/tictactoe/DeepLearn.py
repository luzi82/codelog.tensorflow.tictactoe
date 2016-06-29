import tensorflow as tf

def get_qdn():
    data_in = tf.placeholder(tf.float32, [None,3,3])
    
    mid = data_in
    mid = tf.reshape(mid, [9])
    mid = tf.matmul(mid,tf.zeros([9,10]))
    mid = mid + tf.zeros[10]
    mid = tf.nn.relu(mid)
    mid = tf.matmul(mid,tf.zeros([10,9]))
    mid = mid + tf.zeros[9]
    
    data_out = mid
    
    return data_in,data_out

def get_train(data_out):
    train_ans = tf.placeholder(tf.float32, [None])
    train_actual = tf.placeholder(tf.float32, [None])

    mid = train_ans
    mid = tf.one_hot(mid, 9, axis=-1)
    mid = data_out * mid
    mid = tf.reduce_sum(mid, reduction_indices=[1])
    mid = train_actual - mid
    mid = tf.abs(mid)
    mid = tf.reduce_mean(mid)
    mid = tf.train.GradientDescentOptimizer(0.5).minimize(mid)
    train_step = mid
    
    return train_ans, train_actual, train_step

class DeepLearn(object):

    def __init__(self, params):
        self.data_in, self.data_out = get_qdn()
        self.train_ans, self.train_actual, self.train_step = get_train(self.data_out)
        