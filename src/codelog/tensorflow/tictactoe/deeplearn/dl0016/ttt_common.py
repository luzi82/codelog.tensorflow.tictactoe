from enum import IntEnum

class XY(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Pid(IntEnum):
    O = -1
    X = 1

OPP = {Pid.O: Pid.X,
       Pid.X: Pid.O}

PID_CHAR = {Pid.O:'O',Pid.X:'X',None:' '}

class State(object):
    def __init__(self):
        self.cell_list_list = [[None for _ in range(3)] for _ in range(3) ]

    def __str__(self):
        ret = "|".join("".join( PID_CHAR[cell] for cell in cell_list ) for cell_list in self.cell_list_list)
        return "[{}]".format(ret)

class AbstractLogic(object):

    def get_pid_enum(self):
        return Pid

    def get_new_game_state(self):
        ret = State()
        return ret
