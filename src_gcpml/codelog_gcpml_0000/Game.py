from codelog_gcpml_0000 import Logic as tttl
import six
from builtins import range

PIDCHAR = {tttl.Pid.O:'O',tttl.Pid.X:'X',None:' '}

def printStatus(status, p=six.print_):
#     print("Winner: {}".format(PIDCHAR[status.winner]))
    if status == None:
        p("STATUS_NONE")
    else:
        for cl in status.cell:
            p ("".join(PIDCHAR[c] for c in cl))
        p("Actor: {}".format(PIDCHAR[status.actor]))

class Game(object):

    def __init__(self):
        self.logic = None
        self.playerDict = {}
        self.win_count_dict = {k:0 for k in list(tttl.Pid)}
        self.win_count_dict[None] = 0
        self.bad_move_count_dict = {k:0 for k in list(tttl.Pid)}
        self.game_done_count = 0

    def setPlayer(self,side,player):
        self.playerDict[side] = player

    def run(self,turn_count=None,game_count=None,p=six.print_):
        if turn_count != None:
            for _ in range(turn_count):
                self.turn(p=p)
        elif game_count != None:
            while self.game_done_count < game_count:
                self.turn(p=p)
        else:
            while True:
                self.turn(p=p)

    def turn(self,p = six.print_):
        p('=====')

        status = None if self.logic == None else self.logic.getStatus()
        for _, player in self.playerDict.items():
            player.turn_start(status)

        printStatus(status,p=p)

        if self.logic == None:
            self.logic = tttl.Logic()
            status = self.logic.getStatus()
            for _, player in self.playerDict.items():
                player.new_game(status)
        elif status.actor == None:
            for _, player in self.playerDict.items():
                player.end_game(status)
            self.logic = None
            self.win_count_dict[status.winner] += 1
            self.game_done_count = self.game_done_count + 1
        else :
            activePlayer = self.playerDict[status.actor]
            good = False
            while not good:
                cmd = activePlayer.input(status)
                good = self.logic.action(cmd)
                if not good:
                    p("HLDUXMJC bad action")
                    self.bad_move_count_dict[status.actor] += 1
                    activePlayer.input_error()

        status = None if self.logic == None else self.logic.getStatus()
        for _, player in self.playerDict.items():
            player.turn_end(status)

    def result(self):
        return {
            'win_count_dict':{(k.name if k!=None else "null"):v for k,v in self.win_count_dict.items()},
            'bad_move_count_dict':{k.name:v for k,v in self.bad_move_count_dict.items()},
            'game_done_count':self.game_done_count
        }
