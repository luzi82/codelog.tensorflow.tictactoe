from codelog.tensorflow.tictactoe import Logic as tl
from codelog.tensorflow.tictactoe import Game as tg
import random
from codelog.tensorflow.tictactoe import perfect

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

    def __init__(self):
        pass
    
    def set_side(self,side):
        pass

    def turn_start(self,status):
        pass

    def new_game(self,status):
        pass
    
    def update_status(self,status):
        pass

    def end_game(self,status):
        pass
        
    def input(self,status,retry):
        if retry:
            raise Exception('IOULLYLL')
        available_action_list = []
        for action in ACTION_LIST:
            if status.cell[action.x][action.y] == None:
                available_action_list.append(action)
        return random.choice(available_action_list)

    def turn_end(self,status):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    game = tg.Game()
    
    po = perfect.Player()
    po.set_side(tl.Pid.O)

    px = Player()
    px.set_side(tl.Pid.X)

    game.setPlayer(tl.Pid.O, po)
    game.setPlayer(tl.Pid.X, px)

    game.run(1000)
    print(game.result())
