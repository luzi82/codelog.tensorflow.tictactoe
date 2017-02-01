import tensorflow as tf
from codelog.tensorflow.tictactoe import Game as tg
from codelog.tensorflow.tictactoe import perfect, random_player
from codelog.tensorflow.tictactoe import Logic as tl
from . import deeplearn0020 as dlme
import json, time, os
from builtins import range
import logging
from codelog.tensorflow.tictactoe import py23, util

MY_NAME = os.path.basename(os.path.dirname(__file__))

def vs(player_o,player_x,game_count):
    game = tg.Game()
    
    player_o.set_side(tl.Pid.O)
    player_x.set_side(tl.Pid.X)
    
    game.setPlayer(tl.Pid.O, player_o)
    game.setPlayer(tl.Pid.X, player_x)
    
    game.run(game_count=game_count,p=logging.debug)

    return game.result()

def create_player(param_dict,arg_dict):
    if param_dict['type'] == 'perfect':
        return perfect.Player()
    if param_dict['type'] == 'random':
        return random_player.Player()
    if param_dict['type'] == MY_NAME:
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
    logging.info(vs_dict)
    logging.info(json.dumps(result))
    return result

if __name__ == '__main__':
    timestamp = int(time.time())

    log_filename = os.path.join('log','{}-{}-dlcompare.log'.format(timestamp,MY_NAME))
    py23.makedirs(os.path.dirname(log_filename),exist_ok=True)
    logging.basicConfig(level=logging.INFO,filename=log_filename)

    import argparse

    parser = argparse.ArgumentParser(description='Compare trained model with random and perfect AI')
    parser.add_argument('--path', metavar='N', type=str, nargs='?', default=None, help='working path')
    parser.add_argument('--count', metavar='N', type=int, nargs='?', default=None, help='compare count')
    args = parser.parse_args()
    
    arg_dict = vars(args)

    if arg_dict['path'] == None:
        scan_dir = os.path.join('output',MY_NAME)
        filename_list = os.listdir(scan_dir)
        if len(filename_list) <= 0:
            raise '{} is empty'.format(scan_dir)
        filename_int_list = [util.to_int(filename,-1) for filename in filename_list]
        arg_timestamp = str(max(filename_int_list))
        arg_dict['path'] = os.path.join(scan_dir,str(arg_timestamp))

    if arg_dict['count'] == None:
        arg_dict['count'] = 0
        while os.path.isfile(os.path.join(os.path.join(arg_dict['path'],'deeplearn','sess','{}.index'.format((arg_dict['count']+1)*1000)))):
            arg_dict['count'] += 1

    logging.info('QLCMICZK arg_dict {}'.format(json.dumps(arg_dict)))

    with open(os.path.join(arg_dict['path'],'deeplearn','input_arg_dict.json'),'r') as deeplearn_arg_dict_file:
        deeplearn_arg_dict = json.load(deeplearn_arg_dict_file)

    vs_dict_meta_list = [
        {
            'name':'d','type':MY_NAME,
            'count':arg_dict['count'],'step':1000,
            'filename_format':os.path.join(arg_dict['path'],'deeplearn','sess','{}'),
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
        result = run_vs_dict(vs_dict,deeplearn_arg_dict)
        result_list.append({
            'input':vs_dict,
            'result':result,
        })
    for result in result_list:
        logging.info(json.dumps(result))

    output_filename = os.path.join(arg_dict['path'],'dlcompare','{}.json'.format(timestamp))
    os.makedirs(os.path.dirname(output_filename),exist_ok=True)
    with open(output_filename,'w') as out_file:
        json.dump(result_list,out_file)
#     po = perfect.Player()
#     px = perfect.Player()
#     win_count_dict = vs(po,px,100)
#     print(json.dumps(win_count_dict))
