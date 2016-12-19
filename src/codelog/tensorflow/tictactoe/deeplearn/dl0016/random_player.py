from . import ttt_logic as tl
from . import game as tg
import random

ACTION_LIST = [
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

class Player(object):

    def __init__(self,side):
        self.side = side
        self.logic = tl.Logic()
    
    def turn_start(self,state):
        pass

    def new_game(self,state):
        pass

    def end_game(self,state):
        pass
        
    def input(self,state):
        if self.logic._get_turn_pid(state) != self.side:
            return None
        available_action_list = []
        for action in ACTION_LIST:
            if state.cell_list_list[action.x][action.y] == None:
                available_action_list.append(action)
        return random.choice(available_action_list)

    def input_ok(self):
        pass

    def input_error(self):
        raise Exception('IOULLYLL')

    def turn_end(self,status,continue_,win_dict):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    px = Player(tl.Pid.X)
    po = Player(tl.Pid.O)

    game = tg.Game(tl.Logic(),{tl.Pid.X:px,tl.Pid.O:po})

    game.run(1000)
    print(game.result())
