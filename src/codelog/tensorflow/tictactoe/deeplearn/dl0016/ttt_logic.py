from builtins import range
from .ttt_common import XY, Pid, OPP, AbstractLogic
from .game import Result

class Logic(AbstractLogic):

    def get_continue(self,state):
        for pid in Pid:
            if self._check_win(state,pid):
                return False
        for cell_list in state.cell_list_list:
            for cell in cell_list:
                if cell == None:
                    return True
        return False

    def get_result_dict(self,state):
        for pid in Pid:
            if self._check_win(state,pid):
                return { pid:Result.WIN, OPP[pid]:Result.LOSE }
        for cell_list in state.cell_list_list:
            for cell in cell_list:
                if cell == None:
                    return None
        return { pid:Result.DRAW for pid in Pid }

    def verify_action(self,state, action_dict):
        turn_pid = self._get_turn_pid(state)
        idle_pid = OPP[turn_pid]

        action_valid_dict = {}
        
        turn_action = action_dict[turn_pid]
        if turn_action == None:
            action_valid_dict[turn_pid] = False
        elif state.cell_list_list[turn_action.x][turn_action.y] != None:
            action_valid_dict[turn_pid] = False
        else:
            action_valid_dict[turn_pid] = True

        action_valid_dict[idle_pid] = ( action_dict[idle_pid] == None )

        return action_valid_dict

    def process_action(self,state, action_dict):
        turn_pid = self._get_turn_pid(state)
        turn_action = action_dict[turn_pid]
        state.cell_list_list[turn_action.x][turn_action.y] = turn_pid

    def _check_win(self,state,pid):
        for i in range(3):
            if self._check_combo(state, pid, [XY(i,j) for j in range(3)] ):
                return True
            if self._check_combo(state, pid, [XY(j,i) for j in range(3)] ):
                return True
        if self._check_combo(state, pid, [XY(i,i) for i in range(3)] ):
            return True
        if self._check_combo(state, pid, [XY(i,2-i) for i in range(3)] ):
            return True
        return False

    def _check_combo(self,state,pid,posList):
        for pos in posList:
            if state.cell_list_list[pos.x][pos.y] != pid:
                return False
        return True

    def _get_turn_pid(self,state):
        p_count = {pid:0 for pid in Pid}
        p_count[None] = 0
        for cell_list in state.cell_list_list:
            for cell in cell_list:
                p_count[cell] = p_count[cell] + 1
        if p_count[Pid.X] <= p_count[Pid.O]:
            return Pid.X
        else:
            return Pid.O

if __name__ == '__main__':
    import json
    logic = Logic()

    state = logic.get_new_game_state()
    print(json.dumps(state.__dict__))
    print(json.dumps(logic.get_continue(state)))
    print(json.dumps(logic.get_result_dict(state)))
    print(state)
    assert(logic._get_turn_pid(state)==Pid.X)

    print(json.dumps(logic.verify_action(state,{Pid.X:XY(0,0),Pid.O:None})))
    print(json.dumps(logic.process_action(state,{Pid.X:XY(0,0),Pid.O:None})))
    print(json.dumps(state.__dict__))
    print(json.dumps(logic.get_continue(state)))
    print(json.dumps(logic.get_result_dict(state)))
    print(state)
    assert(logic._get_turn_pid(state)==Pid.O)

    print(json.dumps(logic.verify_action(state,{Pid.X:XY(1,0),Pid.O:XY(0,1)})))
    print(json.dumps(logic.process_action(state,{Pid.X:XY(1,0),Pid.O:XY(0,1)})))
    print(json.dumps(state.__dict__))
    print(json.dumps(logic.get_continue(state)))
    print(json.dumps(logic.get_result_dict(state)))
    print(state)

    print(json.dumps(logic.verify_action(state,{Pid.X:XY(0,0),Pid.O:None})))
    assert(logic.verify_action(state,{Pid.X:XY(0,0),Pid.O:None})[Pid.X] == False)

    print(json.dumps(logic.process_action(state,{Pid.X:XY(1,0),Pid.O:None})))
    print(json.dumps(logic.process_action(state,{Pid.X:None,Pid.O:XY(1,1)})))
    print(json.dumps(logic.process_action(state,{Pid.X:XY(2,0),Pid.O:None})))
    print(json.dumps(state.__dict__))
    print(json.dumps(logic.get_continue(state)))
    print(json.dumps(logic.get_result_dict(state)))
    print(state)
