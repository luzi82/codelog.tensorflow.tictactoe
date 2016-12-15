import tensorflow as tf
from codelog.tensorflow.tictactoe import Game as tg
from codelog.tensorflow.tictactoe import perfect, random_player, dlplayer
from codelog.tensorflow.tictactoe import Logic as tl
from codelog.tensorflow.tictactoe import deeplearn0 as dl0, deeplearn1 as dl1
from codelog.tensorflow.tictactoe import deeplearn5 as dl5
from codelog.tensorflow.tictactoe.deeplearn.dl0010 import deeplearn0010 as dlme
import json, time, os
from builtins import range

PKG_NAME = os.path.basename(os.path.dirname(__file__))

def vs(player_o,player_x,game_count):
    game = tg.Game()
    
    player_o.set_side(tl.Pid.O)
    player_x.set_side(tl.Pid.X)
    
    game.setPlayer(tl.Pid.O, player_o)
    game.setPlayer(tl.Pid.X, player_x)
    
    game.run(game_count=game_count)

    return game.result()

def create_player(param_dict,arg_dict):
    if param_dict['type'] == 'perfect':
        return perfect.Player()
    if param_dict['type'] == 'random':
        return random_player.Player()
    if param_dict['type'] == PKG_NAME:
        dl = dlme.DeepLearn(arg_dict)
        dl.load_sess(param_dict['filename'])
        ret = dlme.DLPlayer(dl)
        ret.set_train_enable(False)
        return ret
    return None

def run_vs_dict(vs_dict,arg_dict):
    tf.reset_default_graph()
    po = create_player(vs_dict['O'],arg_dict)
    px = create_player(vs_dict['X'],arg_dict)
    result = vs(po,px,1000)
    po.close()
    px.close()
    print(vs_dict)
    print(json.dumps(result))
    return result

def to_int(x, y):
    try:
        return int(x)
    except:
        return y

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare trained model with random and perfect AI')
    parser.add_argument('timestamp', metavar='N', type=int, nargs='?', default=None, help='timestamp of training begin')
    args = parser.parse_args()
    arg_dict = vars(args)
    arg_dict['output_path'] = '/tmp/'+PKG_NAME
    arg_dict['random_stddev'] = 0
    arg_dict['random_move_chance'] = 0
    arg_dict['train_beta'] = 0
    arg_dict['train_memory'] = 100

    if arg_dict['timestamp'] == None:
        filename_list = os.listdir('output/'+PKG_NAME)
        if len(filename_list) <= 0:
            raise 'output/'+PKG_NAME+' is empty'
        filename_int_list = [to_int(filename,-1) for filename in filename_list]
        arg_timestamp = str(max(filename_int_list))
    else:
        arg_timestamp = str(arg_dict['timestamp'])

    timestamp = int(time.time())

    vs_dict_meta_list = [
        {
            'name':'d','type':PKG_NAME,
            'count':500,'step':10000,
            'filename_format':'output/'+PKG_NAME+'/'+arg_timestamp+'/{}.ckpt',
        }
    ]

    vs_dict_list = []
    
    vs_dict_list.append({
        'name':'pr',
        'X':{'type':'perfect'},
        'O':{'type':'random'},
    })
    vs_dict_list.append({
        'name':'rp',
        'X':{'type':'random'},
        'O':{'type':'perfect'},
    })
    vs_dict_list.append({
        'name':'rr',
        'X':{'type':'random'},
        'O':{'type':'random'},
    })

    for vs_dict_meta in vs_dict_meta_list:
        for i in range(vs_dict_meta['count']):
            filename = vs_dict_meta['filename_format'].format((i+1)*vs_dict_meta['step'])
            vs_dict_list.append({
                'name':'{}p'.format(vs_dict_meta['name']),'itr':i,
                'X':{'type':vs_dict_meta['type'],'filename':filename},
                'O':{'type':'perfect'},
            })
            vs_dict_list.append({
                'name':'p{}'.format(vs_dict_meta['name']),'itr':i,
                'X':{'type':'perfect'},
                'O':{'type':vs_dict_meta['type'],'filename':filename},
            })
            vs_dict_list.append({
                'name':'{}r'.format(vs_dict_meta['name']),'itr':i,
                'X':{'type':vs_dict_meta['type'],'filename':filename},
                'O':{'type':'random'},
            })
            vs_dict_list.append({
                'name':'r{}'.format(vs_dict_meta['name']),'itr':i,
                'X':{'type':'random'},
                'O':{'type':vs_dict_meta['type'],'filename':filename},
            })

    result_list = []
    for vs_dict in vs_dict_list:
        result = run_vs_dict(vs_dict,arg_dict)
        result_list.append({
            'input':vs_dict,
            'result':result,
        })
    for result in result_list:
        print(json.dumps(result))
    os.makedirs("output/dlcompare",exist_ok=True)
    with open('output/dlcompare/{}.json'.format(timestamp),'w') as out_file:
        json.dump(result_list,out_file)
#     po = perfect.Player()
#     px = perfect.Player()
#     win_count_dict = vs(po,px,100)
#     print(json.dumps(win_count_dict))
