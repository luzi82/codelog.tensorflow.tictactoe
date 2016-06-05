from enum import Enum
import copy

class XY(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Pid(Enum):
    O = -1
    X = 1

OPP = {Pid.O: Pid.X,
       Pid.X: Pid.O}

class Status(object):
    def __init__(self):
        self.winner = None
        self.actor = None
        self.cell = None

class Logic(object):
    def __init__(self):
        self.winner = None
        self.actor = Pid.X
        self.cell = [[None for _ in range(3)] for _ in range(3) ]
    
    def getStatus(self):
        status = Status()
        status.winner = self.winner
        status.actor = self.actor
        status.cell = copy.deepcopy(self.cell)
        return status

    def action(self,pos):
        if self.cell[pos.x][pos.y] != None:
            return False;
        self.cell[pos.x][pos.y] = self.actor
        
        if self.checkWin(self.actor):
            self.winner = self.actor
        
        if self.winner != None:
            self.actor = None
        elif self.checkFull():
            self.actor = None
        else:
            self.actor = OPP[self.actor]

    def checkWin(self,pid):
        for i in range(3):
            if self.checkCombo(pid, [XY(i,j) for j in range(3)] ):
                return True
            if self.checkCombo(pid, [XY(j,i) for j in range(3)] ):
                return True
        if self.checkCombo(pid, [XY(i,i) for i in range(3)] ):
            return True
        if self.checkCombo(pid, [XY(i,2-i) for i in range(3)] ):
            return True

    def checkCombo(self,pid,posList):
        for pos in posList:
            if self.cell[pos.x][pos.y] != pid:
                return False
        return True

    def checkFull(self):
        for cl in self.cell:
            for c in cl:
                if c == None:
                    return False
        return True
