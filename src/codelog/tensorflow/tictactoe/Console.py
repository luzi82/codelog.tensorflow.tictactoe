'''
Created on Jun 5, 2016

@author: luzi82
'''

import codelog.tensorflow.tictactoe.Logic as tttl
from codelog.tensorflow.tictactoe.Game import Game

ACTION_MAP = {
    '1': tttl.XY(2,0),
    '2': tttl.XY(2,1),
    '3': tttl.XY(2,2),
    '4': tttl.XY(1,0),
    '5': tttl.XY(1,1),
    '6': tttl.XY(1,2),
    '7': tttl.XY(0,0),
    '8': tttl.XY(0,1),
    '9': tttl.XY(0,2)
}

PIDCHAR = {tttl.Pid.O:'O',tttl.Pid.X:'X',None:' '}

def printStatus(status):
#     print("Winner: {}".format(PIDCHAR[status.winner]))
    print("Actor: {}".format(PIDCHAR[status.actor]))
    for cl in status.cell:
        print ("".join(PIDCHAR[c] for c in cl))

class ConsolePlayer(object):

    def __init__(self):
        pass

    def set_side(self,side):
        self.side = side

    def new_game(self):
        print("NEW GAME")
    
    def end_game(self,winner):
        print("Winner: {}".format(PIDCHAR[winner]))
        
    def update_status(self,status):
        printStatus(status)
        
    def input(self):
        while True:
            x = input("cmd")
            if x in ACTION_MAP:
                return ACTION_MAP[x]
    
    def input_ok(self,ok):
        if not ok:
            print("NOT OK")

if __name__ == '__main__':
    print("=======")
    game = Game()
    
    po = ConsolePlayer()
    po.set_side(tttl.Pid.O)

    px = ConsolePlayer()
    px.set_side(tttl.Pid.X)
    
    game.setPlayer(tttl.Pid.O, po)
    game.setPlayer(tttl.Pid.X, px)
    
    game.run()
    
#     while True:
#         status = logic.getStatus()
#         printStatus(status)
#         
#         print("=======")
#         if status.actor == None:
#             logic = tttl.Logic()
#         else:
#             x = input("cmd")
#             if x in ACTION_MAP:
#                 ret = logic.action(ACTION_MAP[x])
#                 if not ret:
#                     print("CMD NOT GOOD")
