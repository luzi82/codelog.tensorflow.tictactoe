import codelog.tensorflow.tictactoe.Logic as tttl

class Game(object):

    def __init__(self):
        self.logic = None
        self.playerDict = {}

    def setPlayer(self,side,player):
        self.playerDict[side] = player

    def run(self):
        while True:
            self.turn()

    def turn(self):
        if self.logic == None:
            self.logic = tttl.Logic()
            for _, player in self.playerDict.items():
                player.new_game()
            return

        status = self.logic.getStatus()

        if status.actor == None:
            for _, player in self.playerDict.items():
                player.end_game(status.winner)
            self.logic = None
            return

        activePlayer = self.playerDict[status.actor]
        activePlayer.update_status(status)
        good = False
        while not good:
            cmd = activePlayer.input()
            good = self.logic.action(cmd)
            activePlayer.input_ok(good)

