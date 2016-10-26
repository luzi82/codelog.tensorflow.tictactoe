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
        self.win_count_dict = {k:0 for k in list(tttl.Pid)}
        self.win_count_dict[None] = 0
        self.bad_move_count_dict = {k:0 for k in list(tttl.Pid)}
        self.game_done_count = 0

    def setPlayer(self,side,player):
        self.playerDict[side] = player

    def run(self,turn_count=None,game_count=None):
        if turn_count != None:
            for _ in range(turn_count):
                self.turn()
        elif game_count != None:
            while self.game_done_count < game_count:
                self.turn()
        else:
            while True:
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
            self.win_count_dict[status.winner] += 1
            self.game_done_count = self.game_done_count + 1
            return

        for _, player in self.playerDict.items():
            player.update_status(status)

        activePlayer = self.playerDict[status.actor]
        good = False
        retry = False
        while not good:
            cmd = activePlayer.input(status,retry)
            good = self.logic.action(cmd)
            retry = not good
            if not good:
                print("HLDUXMJC bad action")
                self.bad_move_count_dict[status.actor] += 1

        for _, player in self.playerDict.items():
            player.turn_end()

    def result(self):
        return {
            'win_count_dict':{(k.name if k!=None else "null"):v for k,v in self.win_count_dict.items()},
            'bad_move_count_dict':{k.name:v for k,v in self.bad_move_count_dict.items()},
            'game_done_count':self.game_done_count
        }
