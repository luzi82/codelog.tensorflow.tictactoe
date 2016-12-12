import tensorflow as tf
from codelog.tensorflow.tictactoe import Game as tg
from codelog.tensorflow.tictactoe import perfect, random_player, dlplayer
from codelog.tensorflow.tictactoe import Logic as tl
from codelog.tensorflow.tictactoe import deeplearn0 as dl0, deeplearn1 as dl1
from codelog.tensorflow.tictactoe import deeplearn5 as dl5
from codelog.tensorflow.tictactoe import deeplearn0006
from codelog.tensorflow.tictactoe import deeplearn0007
from codelog.tensorflow.tictactoe import deeplearn0008
import json, time, os

def vs(player_o,player_x,game_count):
    game = tg.Game()
    
    player_o.set_side(tl.Pid.O)
    player_x.set_side(tl.Pid.X)
    
    game.setPlayer(tl.Pid.O, player_o)
    game.setPlayer(tl.Pid.X, player_x)
    
    game.run(game_count=game_count)

    return game.result()

def create_player(param_dict):
    if param_dict['type'] == 'perfect':
        return perfect.Player()
    if param_dict['type'] == 'random':
        return random_player.Player()
    if param_dict['type'] == 'dl0':
        dl = dl0.DeepLearn()
        dl.load_sess(param_dict['filename'])
        ret = dlplayer.DLPlayer(dl)
        ret.set_train_enable(False)
        return ret
    if param_dict['type'] == 'dl1':
        dl = dl1.DeepLearn()
        dl.load_sess(param_dict['filename'])
        ret = dlplayer.DLPlayer(dl)
        ret.set_train_enable(False)
        return ret
    if param_dict['type'] == 'dl5':
        dl = dl5.DeepLearn()
        dl.load_sess(param_dict['filename'])
        ret = dl5.DLPlayer(dl)
        ret.set_train_enable(False)
        return ret
    if param_dict['type'] == 'dl0006':
        dl = deeplearn0006.DeepLearn()
        dl.load_sess(param_dict['filename'])
        ret = deeplearn0006.DLPlayer(dl)
        ret.set_train_enable(False)
        return ret
    if param_dict['type'] == 'dl0007':
        dl = deeplearn0007.DeepLearn()
        dl.load_sess(param_dict['filename'])
        ret = deeplearn0007.DLPlayer(dl)
        ret.set_train_enable(False)
        return ret
    if param_dict['type'] == 'dl0008':
        dl = deeplearn0008.DeepLearn()
        dl.load_sess(param_dict['filename'])
        ret = deeplearn0008.DLPlayer(dl)
        ret.set_train_enable(False)
        return ret
    return None

def run_vs_dict(vs_dict):
    tf.reset_default_graph()
    po = create_player(vs_dict['O'])
    px = create_player(vs_dict['X'])
    result = vs(po,px,1000)
    po.close()
    px.close()
    print(vs_dict)
    print(json.dumps(result))
    return result

if __name__ == '__main__':
    timestamp = int(time.time())

    vs_dict_meta_list = [
        {
            'name':'d','type':'dl0008',
            'count':1000,'step':1000,
            'filename_format':'sess/deeplearn0008/1481217299/{}.ckpt',
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
        result = run_vs_dict(vs_dict)
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
