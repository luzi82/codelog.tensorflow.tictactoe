import tensorflow as tf
from builtins import range
from .ttt_common import XY, Pid, AbstractLogic, State
from .game import Result

INT_TO_PID = [Pid.O, None, Pid.X]
PID_TO_INT = {INT_TO_PID[i]:i for i in range(len(INT_TO_PID))}

INT_TO_RESULT = [Result.LOSE,Result.DRAW,Result.WIN]

BOOL_TO_INT = {False:-1,True:1}

class AbstractDlLogic(AbstractLogic):

    def __init__(self):
        self.ph_state = tf.placeholder(tf.bool, [None,3,3,3])

class Logic(AbstractLogic):

    def get_continue(self,state):
        in_state = self._i_state(state)

        out_continue = self.sess.run([self.t_continue],feed_dict={self.ph_state:[in_state]})
        out_continue = out_continue[0]

        return out_continue

    def get_result_dict(self,state):
        in_state = self._i_state(state)

        out_result = self.sess.run([self.t_result],feed_dict={self.ph_state:[in_state]})
        out_result = out_result[0]

        return { Pid.X:self._o_result(out_result[0]), Pid.O:self._o_result(out_result[1]) }

    def verify_action(self, state, action_dict):
        in_state = self._i_state(state)
        in_action = [self._i_action(action_dict[Pid.X]),self._i_action(action_dict[Pid.O])]

        out_verify_action = self.sess.run([self.t_verify_action],feed_dict={self.ph_state:[in_state],self.ph_action:[in_action]})
        out_verify_action = out_verify_action[0]

        return { Pid.X:out_verify_action[0], Pid.O:out_verify_action[1] }

    def process_action(self, state, action_dict):
        in_state = self._i_state(state)
        in_action = [self._i_action(action_dict[Pid.X]),self._i_action(action_dict[Pid.O])]

        out_state = self.sess.run([self.t_process_action],feed_dict={self.ph_state:[in_state],self.ph_action:[in_action]})
        out_state = out_state[0]

        out_state = self._o_state(out_state)
        self._fill_state(state,out_state)
        
    def _i_state(self,state):
        ret = [[[False for _ in range(3)] for _ in range(3) ] for _ in range(3) ]
        for i in range(3):
            for j in range(3):
                ret[i][j][PID_TO_INT[state.cell_list_list[i][j]]] = True
        return ret

    def _i_action(self,action):
        if action == None:
            return 9
        return action.x * 3 + action.y

    def _o_result(self,v):
        return INT_TO_RESULT[v]

    def _o_state(self,s):
        state = State()
        for i in range(3):
            for j in range(3):
                state.cell_list_list[i][j] = INT_TO_PID[s[i][j]]
        return state

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
