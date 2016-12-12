# clone of deeplearn 6
# add random in decision weight

import tensorflow as tf
from collections import deque 
import json
import time, os
import copy
from codelog.tensorflow.tictactoe import dlplayer
from codelog.tensorflow.tictactoe.Game import Game
import codelog.tensorflow.tictactoe.Logic as tttl
import sys
import random

MY_FILENAME = (os.path.basename(sys.modules['__main__'].__file__))
MY_NAME = MY_FILENAME[:-3] # remove .py

OUTPUT_COUNT = 9
RANDOM_STDDEV = 0.1
RANDOM_MOVE_CHANCE = 0.05

def new_state_ph():
    return tf.placeholder(tf.float32, [None,3,3])

def new_var_dict():
    stddev = 0.4
    ret = {}
    ret['w0']=tf.Variable(tf.random_normal([9,50] ,stddev=stddev,dtype=tf.float32))
    ret['b1']=tf.Variable(tf.random_normal([50]   ,stddev=stddev,dtype=tf.float32))
    ret['w2']=tf.Variable(tf.random_normal([50,50],stddev=stddev,dtype=tf.float32))
    ret['b3']=tf.Variable(tf.random_normal([50]   ,stddev=stddev,dtype=tf.float32))
    ret['w4']=tf.Variable(tf.random_normal([50,9] ,stddev=stddev,dtype=tf.float32))
    ret['b5']=tf.Variable(tf.random_normal([9]    ,stddev=stddev,dtype=tf.float32))
    return ret

def new_train_ph_dict():
    ret = {}
    ret['state_0']=new_state_ph()
    ret['choice_0']=tf.placeholder(tf.int32,[None])
    ret['reward_1']=tf.placeholder(tf.float32,[None])
    ret['cont']=tf.placeholder(tf.float32,[None])
    ret['state_1']=new_state_ph()
    return ret

def var_dict_to_ph_dict(var_dict):
    ret = {}
    for k in var_dict:
        ret[k] = tf.placeholder(tf.float32, var_dict[k].get_shape())
    return ret

def get_q(state_ph,var_dict):
    mid = state_ph
    mid = tf.reshape(mid, [-1,9])
    mid = tf.matmul(mid,var_dict['w0'])
    mid = mid + var_dict['b1']
    mid = tf.nn.elu(mid)
    mid = tf.matmul(mid,var_dict['w2'])
    mid = mid + var_dict['b3']
    mid = tf.nn.elu(mid)
    mid = tf.matmul(mid,var_dict['w4'])
    mid = mid + var_dict['b5'] * tf.constant(0.05)
    return mid

def get_train_choice(state_ph,var_dict,random_t,mask):
    score = get_q(state_ph,var_dict)
    mid = score
    mid = mid + random_t
    mid = tf.exp(mid)
    mid = mid * mask
    mid = tf.argmax(mid, dimension=1)
    choice = mid
    return score, choice

TRAIN_BETA = 0.99
ELEMENT_L2_FACTOR = 10.0
L2_WEIGHT = 0.1

def get_l2(m):
    return tf.reduce_sum(m*m)

def get_train(train_ph_dict,var_dict,var_ph_dict):
    mid0 = tf.one_hot(train_ph_dict['choice_0'], 9, axis=-1, dtype=tf.float32)
    mid0 = mid0 * get_q(train_ph_dict['state_0'],var_dict)
    mid0 = tf.reduce_sum(mid0, reduction_indices=[1])

    mid1 = get_q(train_ph_dict['state_1'],var_ph_dict)
    mid1 = tf.reduce_max(mid1, reduction_indices=[1])  
    mid1 = mid1 * train_ph_dict['cont']
    mid1 = mid1 * tf.constant(TRAIN_BETA)

    l2r = tf.constant(0.0)
    cell_count = tf.constant(0.0)
    for v in var_dict.values():
        l2r = l2r + get_l2(v)
        cell_count = cell_count + tf.to_float(tf.size(v))
    l2r = l2r / cell_count
    l2r = l2r / tf.constant(ELEMENT_L2_FACTOR*ELEMENT_L2_FACTOR)
    l2r = l2r * tf.constant(L2_WEIGHT)
    
    mid = mid0+mid1-train_ph_dict['reward_1']
#    mid = mid * mid
    mid = tf.abs(mid)
    mid = tf.reduce_mean(mid)
    score_diff = mid
    mid = mid + l2r
    mid = mid + ( tf.abs( tf.reduce_mean(var_dict['b5']) ) * tf.constant(L2_WEIGHT) )

    loss = mid

    mid = tf.train.AdamOptimizer().minimize(mid,var_list=var_dict.values())
    train = mid
    
    return train, loss, score_diff

TRAIN_MEMORY = 20000

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
        self.random_t = tf.random_normal([OUTPUT_COUNT], stddev=RANDOM_STDDEV)
        self.mask = tf.placeholder(tf.float32, [None,9])
        self.var_ph_dict = var_dict_to_ph_dict(self.var_dict)

        # choice
        self.choice_state = new_state_ph()
        self.score, self.train_choice = get_train_choice(self.choice_state,self.var_dict,self.random_t,self.mask)
        
        # train
        self.train_ph_dict = new_train_ph_dict()
        self.train, self.loss, self.score_diff = get_train(self.train_ph_dict,self.var_dict,self.var_ph_dict)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        
        self.train_count = 0
        self.saver = tf.train.Saver(self.var_dict,max_to_keep=None)
        self.timestamp = int(time.time())
        
    def load_sess(self,filename):
        self.saver.restore(self.sess, filename)

    def cal_choice(self, state_0, mask, train_enable):
        if train_enable and random.random() < RANDOM_MOVE_CHANCE:
            choice_0 = random.randrange(OUTPUT_COUNT)
            print("GDNZVUUU cal_choice rand "+str(choice_0))
            return {
                'state_0': state_0,
                'choice_0': choice_0,
                'state_1': None,
                'cont': None,
                'reward_1': None,
            }, None
        else:
            score, choice_0 = self.sess.run([self.score, self.train_choice],feed_dict={self.choice_state:[state_0],self.mask:[mask]})
            score = score[0].tolist()
            choice_0 = choice_0.tolist()[0]
            print("PWWRJCYQ pred "+json.dumps([int(x*100) for x in score]))
            return {
                'state_0': state_0,
                'choice_0': choice_0,
                'state_1': None,
                'cont': None,
                'reward_1': None,
            }, score[choice_0]

    def push_train_dict(self, train_dict):
        print("EECSQBUX push_train_dict: "+json.dumps(train_dict))
        for k, v in self.queue.items():
            v.append(train_dict[k])

    def do_train(self):
        if len(self.queue['state_0']) < 100:
            return
        feed_dict = {}
        for k in self.train_ph_dict:
            feed_dict[self.train_ph_dict[k]] = list(self.queue[k])
        for k in self.var_ph_dict:
            feed_dict[self.var_ph_dict[k]] = self.var_dict[k].eval(self.sess)
        _, loss, score_diff = self.sess.run([self.train,self.loss,self.score_diff],feed_dict=feed_dict)
        print('ZPDDPYFD loss '+str(loss)+' '+str(score_diff))
        self.train_count += 1
        if self.train_count % 1000 == 0:
            os.makedirs("sess/{}/{}".format(MY_NAME,self.timestamp),exist_ok=True)
            self.saver.save(self.sess,"sess/{}/{}/{}.ckpt".format(MY_NAME,self.timestamp,self.train_count))
        print('HZQQMSQT '+MY_NAME+' '+str(self.train_count))

    def close(self):
        self.sess.close()

REWARD_WIN = 1
REWARD_LOSE = -1
REWARD_DRAW = 0
REWARD_STEP = -0.05
REWARD_BAD = -2

class DLPlayer(object):

    def __init__(self, dl):
        self.dl = dl
        self.train_enable = True
    
    def set_side(self,side):
        self.side = side
        
    def turn_start(self,status):
        self.legit_mask = None
        self.train_dict = None
        self.last_choice = None

    def new_game(self,status):
        self.legit_mask = None
        self.train_dict = None
        self.last_choice = None
    
    def set_train_enable(self,train_enable):
        self.train_enable = train_enable
    
    def update_status(self,status):
        pass

    def end_game(self,status):
        self.legit_mask = None
        self.train_dict = None
        self.last_choice = None

    def input(self,status):
        if self.legit_mask == None:
            self.legit_mask = [1.0]*9

        new_status = dlplayer.conv_status(status,self.side)
        self.train_dict, _ = self.dl.cal_choice(new_status,self.legit_mask,self.train_enable)
        
        choice = self.train_dict['choice_0']
        print("GKPMPCLI choice: "+str(choice))

        self.last_choice = choice
        return dlplayer.ACTION_MAP[choice]

    def input_error(self):
        self.legit_mask[self.last_choice] = 0.0
        if self.train_enable:
            train_dict = copy.copy(self.train_dict)
            train_dict['state_1'] = train_dict['state_0']
            train_dict['cont'] = 0
            train_dict['reward_1'] = REWARD_BAD
            self.dl.push_train_dict(train_dict)
            self.dl.do_train()

    def turn_end(self,status):
        if self.train_enable:
            if self.last_choice != None:
                if status.actor == None:
                    reward = REWARD_WIN if status.winner == self.side \
                        else REWARD_DRAW if status.winner == None \
                        else REWARD_LOSE
                    train_dict = copy.copy(self.train_dict)
                    train_dict['state_1'] = train_dict['state_0']
                    train_dict['cont'] = 0
                    train_dict['reward_1'] = reward
                    self.dl.push_train_dict(train_dict)
                    self.dl.do_train()
                else:
                    new_status = dlplayer.conv_status(status,tttl.OPP[self.side])
                    train_dict = copy.copy(self.train_dict)
                    train_dict['state_1'] = new_status
                    train_dict['cont'] = 1
                    train_dict['reward_1'] = REWARD_STEP
                    self.dl.push_train_dict(train_dict)
                    self.dl.do_train()

        self.legit_mask = None
        self.train_dict = None
        self.last_choice = None

    def close(self):
        self.legit_mask = None
        self.train_dict = None
        self.last_choice = None
        self.dl.close()

if __name__ == '__main__':
    game = Game()
    dl = DeepLearn()
    
    po = DLPlayer(dl)
    po.set_side(tttl.Pid.O)

    px = DLPlayer(dl)
    px.set_side(tttl.Pid.X)
    
    game.setPlayer(tttl.Pid.O, po)
    game.setPlayer(tttl.Pid.X, px)
    
    game.run()
