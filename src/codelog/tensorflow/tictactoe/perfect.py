from codelog.tensorflow.tictactoe import Logic as tl

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

# choice, must_win, may_win, must_lose, may_lose
def cal_min_max(status,actor):
    ret = None
    
    if status.actor == None:
        if status.winner == actor:
            return None, True, True, False, False
        elif status.winner == None:
            return None, False, False, False, False
        else:
            return None, False, False, True, True
    
    must_win = True
    may_win = False
    must_lose = True
    may_lose = False
    move_score = -999
    move_list = None

    for action in ACTION_DICT:
        logic = tl.Logic()
        logic.set_status(status)
        if not logic.action(action):
            continue
        
    
    return move_list, must_win, may_win, must_lose, may_lose

class Player(object):

    def __init__(self, dl):
        self.dl = dl
    
    def set_side(self,side):
        self.side = side

    def new_game(self,status):
        pass
    
    def end_game(self,status):
        pass
        
    def input(self,status,retry):
        return cal_min_max(status)
