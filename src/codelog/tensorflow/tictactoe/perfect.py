from codelog.tensorflow.tictactoe import Logic as tl
from codelog.tensorflow.tictactoe import Game as tg
import random, json

ACTION_DICT = [
    tl.XY(0,0),
    tl.XY(0,1),
    tl.XY(0,2),
    tl.XY(1,0),
    tl.XY(1,1),
    tl.XY(1,2),
    tl.XY(2,0),
    tl.XY(2,1),
    tl.XY(2,2),
]

CAL_MIN_MAX_DICT={}

def status_key(status,actor):
    ret = []
    for cl in status.cell:
        for c in cl:
            ret.append(int(c)if c != None else 0)
    ret.append(int(actor))
    ret = json.dumps(ret)
    return ret

# choice, score
def cal_min_max(status,actor):
    k = status_key(status,actor)
    if k in CAL_MIN_MAX_DICT:
        return CAL_MIN_MAX_DICT[k][0], CAL_MIN_MAX_DICT[k][1]
    
    #tg.printStatus(status)
    if status.actor == None:
        if status.winner == actor:
            return [], 1
        elif status.winner == None:
            return [], 0
        else:
            return [], -1
    
    ret_score = -999
    ret_choice_list = None

    logic = tl.Logic()

    for action in ACTION_DICT:
        logic.set_status(status)
        if not logic.action(action):
            continue
        new_status = logic.getStatus()
        _, s = cal_min_max(new_status,tl.OPP[actor])
        s *= -1
        if s > ret_score:
            ret_score = s
            ret_choice_list = [action]
        elif s == ret_score:
            ret_choice_list.append(action)
    
    CAL_MIN_MAX_DICT[k] = [ret_choice_list,ret_score]
    
    return ret_choice_list, ret_score

class Player(object):

    def __init__(self):
        self.side = None
    
    def set_side(self,side):
        self.side = side

    def turn_start(self,status):
        pass

    def new_game(self,status):
        pass

    def update_status(self,status):
        pass

    def end_game(self,status):
        if status.winner == tl.OPP[self.side]:
            raise Exception('FTIRAQKFKT')
        
    def input(self,status,retry):
        if retry:
            raise Exception('FLSTQLRK')
        choice_list, _ = cal_min_max(status,self.side)
        return random.choice(choice_list)

    def turn_end(self,status):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    game = tg.Game()
    
    po = Player()
    po.set_side(tl.Pid.O)

    px = Player()
    px.set_side(tl.Pid.X)
    
    game.setPlayer(tl.Pid.O, po)
    game.setPlayer(tl.Pid.X, px)
    
    game.run(-1)
