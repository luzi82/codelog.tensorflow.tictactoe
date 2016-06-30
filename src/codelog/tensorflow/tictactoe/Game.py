import codelog.tensorflow.tictactoe.Logic as tttl

PIDCHAR = {tttl.Pid.O:'O',tttl.Pid.X:'X',None:' '}

def printStatus(status):
#     print("Winner: {}".format(PIDCHAR[status.winner]))
    for cl in status.cell:
        print ("".join(PIDCHAR[c] for c in cl))
    print("Actor: {}".format(PIDCHAR[status.actor]))

class Game(object):

    def __init__(self):
        self.logic = None
        self.playerDict = {}

    def setPlayer(self,side,player):
        self.playerDict[side] = player

    def run(self,t):
        if t < 0:
            while True:
                self.turn()
        else:
            for _ in range(t):
                self.turn()

    def turn(self):
        print('=====')
        if self.logic == None:
            self.logic = tttl.Logic()
            status = self.logic.getStatus()
            for _, player in self.playerDict.items():
                player.new_game(status)
            return

        status = self.logic.getStatus()
        printStatus(status)

        if status.actor == None:
            for _, player in self.playerDict.items():
                player.end_game(status)
            self.logic = None
            return

        activePlayer = self.playerDict[status.actor]
        good = False
        retry = False
        while not good:
            cmd = activePlayer.input(status,retry)
            good = self.logic.action(cmd)
            retry = not good
            if not good:
                print("HLDUXMJC bad action")

