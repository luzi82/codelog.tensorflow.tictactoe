'''
Created on 29 Jun 2016

@author: luzi
'''
import copy
from codelog.tensorflow.tictactoe.Game import Game
from codelog.tensorflow.tictactoe import deeplearn0 as deeplearn
import codelog.tensorflow.tictactoe.Logic as tttl

def conv_status(status,side):
    return [[1 if c == side else 0 if c == None else -1 for c in cl] for cl in status.cell]

ACTION_MAP = [
    tttl.XY(0,0),
    tttl.XY(0,1),
    tttl.XY(0,2),
    tttl.XY(1,0),
    tttl.XY(1,1),
    tttl.XY(1,2),
    tttl.XY(2,0),
    tttl.XY(2,1),
    tttl.XY(2,2),
]

REWARD_WIN = 0.9
REWARD_LOSE = -0.9
REWARD_DRAW = 0
REWARD_STEP = -0.001
REWARD_BAD = -1

class DLPlayer(object):

    def __init__(self, dl):
        self.dl = dl
        self.train_enable = True
    
    def set_side(self,side):
        self.side = side

    def new_game(self,status):
        self.train_dict = None
    
    def set_train_enable(self,train_enable):
        self.train_enable = train_enable
    
    def end_game(self,status):
        if self.train_enable:
            train_dict = copy.copy(self.train_dict)
            train_dict['state_1'] = conv_status(status,self.side)
            train_dict['cont'] = 0
            train_dict['reward_1'] = REWARD_WIN if status.winner == self.side else REWARD_DRAW if status.winner == None else REWARD_LOSE
            self.dl.push_train_dict(train_dict)
            self.dl.do_train()
            self.train_dict = None
        
    def input(self,status,retry):
        if not retry:
            self.mask = [1.0]*9
        else:
            self.mask[self.train_dict['choice_0']] = 0.0
        new_status = conv_status(status,self.side)
        if self.train_dict != None:
            train_dict = copy.copy(self.train_dict)
            train_dict['state_1'] = new_status
            train_dict['cont'] = 0 if retry else 1
            train_dict['reward_1'] = REWARD_BAD if retry else REWARD_STEP
            self.dl.push_train_dict(train_dict)
            self.dl.do_train()
        self.train_dict, _ = self.dl.cal_choice(new_status,self.mask)
        print("GKPMPCLI choice: "+str(self.train_dict['choice_0']))
#         use_chance = score + 1.0
#         use_chance = min(use_chance,0.95)
#         use_chance = 0.95
#         if random.random() > use_chance:
#             self.train_dict['choice_0'] = random.randrange(9)
#             print("KEKPIAXP random: "+str(self.train_dict['choice_0']))
        return ACTION_MAP[self.train_dict['choice_0']]

if __name__ == '__main__':
    game = Game()
    dl = deeplearn.DeepLearn()
    
    po = DLPlayer(dl)
    po.set_side(tttl.Pid.O)

    px = DLPlayer(dl)
    px.set_side(tttl.Pid.X)
    
    game.setPlayer(tttl.Pid.O, po)
    game.setPlayer(tttl.Pid.X, px)
    
    game.run(-1)
