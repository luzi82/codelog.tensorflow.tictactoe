import six
from builtins import range
from enum import IntEnum

class Result(IntEnum):
    WIN = 1
    DRAW = 0
    LOSE = -1

class Game(object):

    def __init__(self,logic,player_dict):
        self.logic = logic
        self.player_dict = player_dict

        self.pid_enum = logic.get_pid_enum()

        self.result_dict_dict = { pid:{ r:0 for r in Result } for pid in self.pid_enum }
        self.bad_move_count_dict = { pid:0 for pid in self.pid_enum }
        self.game_done_count = 0

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

        for player in self.player_dict.values():
            player.turn_start(self.state)

        p("status: {}".format(self.state))

        if self.state == None:
            self.state = self.logic.get_new_game_state()
            for player in self.player_dict.values():
                player.new_game(self.state)
        elif not self.logic.get_continue(self.state):
            result_dict = self.logic.get_result_dict(self.state)
            for pid in self.pid_enum:
                self.result_dict_dict[pid][result_dict[pid]] += 1
            for player in self.player_dict.values():
                player.end_game(self.state)
            self.state = None
            self.game_done_count += 1
        else:
            good_dict = {pid:False for pid in self.pid_enum}
            action_dict = {pid:None for pid in self.pid_enum}
            while True:
                for pid in self.pid_enum:
                    if good_dict[pid]:
                        continue
                    action_dict[pid] = self.player_dict[pid].input(self.state)
                
                good_dict_0 = self.logic.verify_action(self.state,action_dict)
                
                for pid in self.pid_enum:
                    if good_dict[pid]:
                        continue
                    if good_dict_0[pid]:
                        self.player_dict[pid].input_ok()
                    else:
                        self.bad_move_count_dict[pid] += 1
                        self.player_dict[pid].input_error()
                
                good_dict = good_dict_0
                
                for pid in self.pid_enum:
                    if not good_dict[pid]:
                        continue
                break
            self.logic.process_action(self.state,action_dict)

        if self.state == None:
            continue_ = None
            win_dict = None
        else:
            continue_ = self.logic.get_continue(self.state)
            win_dict = self.logic.get_result_dict(self.state)

        for player in self.player_dict.values():
            player.turn_end(self.state,continue_,win_dict)

    def result(self):
        return {
            'result_dict_dict':{k.name:{kk.name:v for kk,v in vl.items()} for k,vl in self.result_dict_dict.items()},
            'bad_move_count_dict':{k.name:v for k,v in self.bad_move_count_dict.items()},
            'game_done_count':self.game_done_count
        }
