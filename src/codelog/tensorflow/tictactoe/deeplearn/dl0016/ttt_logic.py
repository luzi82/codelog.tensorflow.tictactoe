from enum import IntEnum
import copy
from builtins import range

class XY(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Pid(IntEnum):
    O = -1
    X = 1

OPP = {Pid.O: Pid.X,
       Pid.X: Pid.O}

class State(object):
    def __init__(self):
        self.cell_list_list = None

    @staticmethod
    def new_game():
        ret = State()
        ret.cell_list_list = [[None for _ in range(3)] for _ in range(3) ]
        return ret

class Logic(object):
    def __init__(self):
        self.state = None
    
    def new_game(self):
        self.state = State.new_game()
    
    def get_state(self):
        state = copy.deepcopy(self.state)
        return state
    
    def set_state(self,state):
        self.state = copy.deepcopy(state)
    
    def get_continue(self):
        win_dict = self.get_win_dict()
        for v in win_dict.values():
            if v:
                return False
        for cell_list in self.state.cell_list_list:
            for cell in cell_list:
                if cell == None:
                    return True
        return False

    def get_win_dict(self):
        ret = { pid: self.check_win(pid) for pid in Pid }
        return ret

    def check_win(self,pid):
        for i in range(3):
            if self._check_combo(pid, [XY(i,j) for j in range(3)] ):
                return True
            if self._check_combo(pid, [XY(j,i) for j in range(3)] ):
                return True
        if self._check_combo(pid, [XY(i,i) for i in range(3)] ):
            return True
        if self._check_combo(pid, [XY(i,2-i) for i in range(3)] ):
            return True
        return False

    def action(self,action_dict):
        turn_pid = self._get_turn_pid()
        idle_pid = OPP[turn_pid]

        action_valid_dict = {}
        
        turn_action = action_dict[turn_pid]
        if turn_action == None:
            action_valid_dict[turn_pid] = False
        elif self.state.cell_list_list[turn_action.x][turn_action.y] != None:
            action_valid_dict[turn_pid] = False
        else:
            self.state.cell_list_list[turn_action.x][turn_action.y] = turn_pid
            action_valid_dict[turn_pid] = True

        action_valid_dict[idle_pid] = ( action_dict[idle_pid] == None )

        return action_valid_dict

    def _check_combo(self,pid,posList):
        for pos in posList:
            if self.state.cell_list_list[pos.x][pos.y] != pid:
                return False
        return True

    def _get_turn_pid(self):
        p_count = {pid:0 for pid in Pid}
        p_count[None] = 0
        for cell_list in self.state.cell_list_list:
            for cell in cell_list:
                p_count[cell] = p_count[cell] + 1
        if p_count[Pid.X] <= p_count[Pid.O]:
            return Pid.X
        else:
            return Pid.O

if __name__ == '__main__':
    import json
    logic = Logic()

    logic.new_game()
    print(json.dumps(logic.get_state().__dict__))
    print(json.dumps(logic.get_continue()))
    print(json.dumps(logic.get_win_dict()))
    assert(logic._get_turn_pid()==Pid.X)

    print(json.dumps(logic.action({Pid.X:XY(0,0),Pid.O:None})))
    print(json.dumps(logic.get_state().__dict__))
    print(json.dumps(logic.get_continue()))
    print(json.dumps(logic.get_win_dict()))
    assert(logic._get_turn_pid()==Pid.O)

    print(json.dumps(logic.action({Pid.X:XY(1,0),Pid.O:XY(0,1)})))
    print(json.dumps(logic.get_state().__dict__))
    print(json.dumps(logic.get_continue()))
    print(json.dumps(logic.get_win_dict()))

    print(json.dumps(logic.action({Pid.X:XY(1,0),Pid.O:None})))
    print(json.dumps(logic.action({Pid.X:None,Pid.O:XY(1,1)})))
    print(json.dumps(logic.action({Pid.X:XY(2,0),Pid.O:None})))
    print(json.dumps(logic.get_state().__dict__))
    print(json.dumps(logic.get_continue()))
    print(json.dumps(logic.get_win_dict()))
