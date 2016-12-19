import six
from builtins import range

class Game(object):

    def __init__(self,logic,player_dict):
        self.logic = logic
        self.player_dict = player_dict

        self.pid_enum = logic.get_pid_enum()

        self.win_count_dict = {pid:0 for pid in self.pid_enum}
        self.win_count_dict[None] = 0
        self.bad_move_count_dict = {pid:0 for pid in self.pid_enum}

        self.state = None

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

        for player in self.playerDict.values():
            player.turn_start(self.state)

        p("status: {}".format(self.state))

        if self.state == None:
            self.state = self.logic.get_new_game_state()
            for player in self.playerDict.values():
                player.new_game(self.state)
        elif not self.logic.get_continue(self.state):
            for player in self.playerDict.values():
                player.end_game(self.state)
            self.state = None
        else:
            good_dict = {pid:False for pid in self.pid_enum}
            action_dict = {pid:None for pid in self.pid_enum}
            while True:
                for pid in self.pid_enum:
                    if good_dict[pid]:
                        continue
                    action_dict[pid] = self.player_dict[pid].input(self.state)
                
                good_dict_0 = self.logic.verify_action(action_dict)
                
                for pid in self.pid_enum:
                    if good_dict[pid]:
                        continue
                    if good_dict_0[pid]:
                        self.player_dict[pid].input_ok()
                    else:
                        self.player_dict[pid].input_error()
                
                good_dict = good_dict_0
                
                for pid in self.pid_enum:
                    if not good_dict[pid]:
                        continue
                break

        for player in self.playerDict.values():
            player.turn_end(self.status)

    def result(self):
        return {
            'win_count_dict':{(k.name if k!=None else "null"):v for k,v in self.win_count_dict.items()},
            'bad_move_count_dict':{k.name:v for k,v in self.bad_move_count_dict.items()},
            'game_done_count':self.game_done_count
        }
