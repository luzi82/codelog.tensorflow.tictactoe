'''
Created on Jun 5, 2016

@author: luzi82
'''

import codelog.tensorflow.tictactoe.Logic as tttl

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
    print("Winner: {}".format(PIDCHAR[status.winner]))
    print("Actor: {}".format(PIDCHAR[status.actor]))
    for cl in status.cell:
        print ("".join(PIDCHAR[c] for c in cl))

if __name__ == '__main__':
    print("=======")
    logic = tttl.Logic()
    
    while True:
        status = logic.getStatus()
        printStatus(status)
        
        print("=======")
        if status.actor == None:
            logic = tttl.Logic()
        else:
            x = input("cmd")
            if x in ACTION_MAP:
                logic.action(ACTION_MAP[x])
