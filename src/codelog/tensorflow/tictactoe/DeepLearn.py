import tensorflow as tf
from collections import deque 
import json

def new_state_ph():
    return tf.placeholder(tf.float32, [None,3,3])

def new_var_dict():
    ret = {}
    ret['w0']=tf.Variable(tf.zeros([9,50],dtype=tf.float32))
    ret['b1']=tf.Variable(tf.zeros([50],dtype=tf.float32))
    ret['w2']=tf.Variable(tf.zeros([50,9],dtype=tf.float32))
    ret['b3']=tf.Variable(tf.zeros([9],dtype=tf.float32))
    return ret

def new_train_ph_dict():
    ret = {}
    ret['state_0']=new_state_ph()
    ret['choice_0']=tf.placeholder(tf.int32,[None])
    ret['reward_1']=tf.placeholder(tf.float32,[None])
    ret['cont']=tf.placeholder(tf.float32,[None])
    ret['state_1']=new_state_ph()
    return ret

def get_q(state_ph,var_dict):
    mid = state_ph
    mid = tf.reshape(mid, [-1,9])
    mid = tf.matmul(mid,var_dict['w0'])
    mid = mid + var_dict['b1']
    mid = tf.nn.relu(mid)
    mid = tf.matmul(mid,var_dict['w2'])
    mid = mid + var_dict['b3']
    return mid

def get_choice(state_ph,var_dict):
    score = get_q(state_ph,var_dict)
    choice = tf.argmax(score, dimension=1)
    return score, choice

TRAIN_BETA = 0.99

def get_train(train_ph_dict,var_dict):
    mid0 = tf.one_hot(train_ph_dict['choice_0'], 9, axis=-1, dtype=tf.float32)
    mid0 = mid0 * get_q(train_ph_dict['state_0'],var_dict)
    mid0 = tf.reduce_sum(mid0, reduction_indices=[1])

    mid1 = get_q(train_ph_dict['state_1'],var_dict)
    mid1 = tf.reduce_max(mid1, reduction_indices=[1])  
    mid1 = mid1 * train_ph_dict['cont']
    mid1 = mid1 * tf.constant(TRAIN_BETA)
    
    mid = mid0-mid1-train_ph_dict['reward_1']
    mid = tf.abs(mid)
    mid = tf.reduce_mean(mid)
    mid = tf.train.GradientDescentOptimizer(0.5).minimize(mid)
    
    return mid

TRAIN_MEMORY = 1000

class DeepLearn(object):
    
    def __init__(self):
        self.var_dict = new_var_dict()
        self.queue = {
            'state_0': deque(maxlen=TRAIN_MEMORY),
            'choice_0': deque(maxlen=TRAIN_MEMORY),
            'state_1': deque(maxlen=TRAIN_MEMORY),
            'cont': deque(maxlen=TRAIN_MEMORY),
            'reward_1': deque(maxlen=TRAIN_MEMORY),
        }

        # choice
        self.choice_state = new_state_ph()
        self.score, self.choice = get_choice(self.choice_state,self.var_dict)
        
        # train
        self.train_ph_dict = new_train_ph_dict()
        self.train = get_train(self.train_ph_dict,self.var_dict)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def cal_choice(self, state_0):
        score, choice_0 = self.sess.run([self.score, self.choice],feed_dict={self.choice_state:[state_0]})
        print(json.dumps([int(x*100) for x in score[0].tolist()]))
        choice_0 = choice_0[0]
        return {
            'state_0': state_0,
            'choice_0': choice_0,
            'state_1': None,
            'cont': None,
            'reward_1': None,
        }

    def push_train_dict(self, train_dict):
        for k, v in self.queue.items():
            v.append(train_dict[k])

    def trainn(self):
        feed_dict={
            self.train_ph_dict['state_0']: list(self.queue['state_0']),
            self.train_ph_dict['choice_0']: list(self.queue['choice_0']),
            self.train_ph_dict['state_1']: list(self.queue['state_1']),
            self.train_ph_dict['cont']: list(self.queue['cont']),
            self.train_ph_dict['reward_1']: list(self.queue['reward_1']),
        }
        self.sess.run(self.train,feed_dict=feed_dict)
