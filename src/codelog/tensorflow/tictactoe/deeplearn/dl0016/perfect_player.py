from . import ttt_logic as tl
from . import game as tg
import random, json
import copy

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
    for cl in status.cell_list_list:
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
    
    if not LOGIC.get_continue(status):
        result = LOGIC.get_result_dict(status)[actor]
        if result == tg.Result.WIN:
            return [], 1
        elif result == tg.Result.DRAW:
            return [], 0
        else:
            return [], -1
    
    ret_score = -999
    ret_choice_list = None

    for action in ACTION_DICT:
        if status.cell_list_list[action.x][action.y] != None:
            continue
        new_status = copy.deepcopy(status)
        LOGIC.process_action(new_status,{actor:action,tl.OPP[actor]:None})
        _, s = cal_min_max(new_status,tl.OPP[actor])
        s *= -1
        if s > ret_score:
            ret_score = s
            ret_choice_list = [action]
        elif s == ret_score:
            ret_choice_list.append(action)
    
    CAL_MIN_MAX_DICT[k] = [ret_choice_list,ret_score]
    
    return ret_choice_list, ret_score

LOGIC = tl.Logic()

class Player(object):

    def __init__(self,side):
        self.side = side

    def turn_start(self,state):
        pass

    def new_game(self,state):
        pass

    def end_game(self,state):
        if LOGIC.get_result_dict(state)[self.side] == tg.Result.LOSE:
            raise Exception('FTIRAQKFKT')

    def input(self,state):
        if LOGIC._get_turn_pid(state) != self.side:
            return None
        choice_list, _ = cal_min_max(state,self.side)
        return random.choice(choice_list)
    
    def input_ok(self):
        pass

    def input_error(self):
        raise Exception('FLSTQLRK')

    def turn_end(self,state,continue_,win_dict):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    px = Player(tl.Pid.X)
    po = Player(tl.Pid.O)

    game = tg.Game(tl.Logic(),{tl.Pid.X:px,tl.Pid.O:po})

    game.run(10000)
    print(game.result())
