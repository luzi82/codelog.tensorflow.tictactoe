'''
Created on Jul 12, 2016

@author: luzi82
'''

import json
import matplotlib.pyplot as plt
from builtins import range
import os.path
import argparse
from codelog.tensorflow.tictactoe import util as myutil

# ITR_COUNT = 50
MY_NAME = os.path.basename(os.path.dirname(__file__))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show graph')
    parser.add_argument('filename', metavar='N', type=int, nargs='?', default=None, help='compare filename')
    args = parser.parse_args()

    arg_dict = vars(args)
    
    if arg_dict['filename'] == None:
        dl_path = os.path.join('output',MY_NAME)
        
        dl_filename_list = os.listdir(dl_path)
        if len(dl_filename_list) <= 0:
            raise '{} is empty'.format(dl_path)
        dl_filename_int_list = [myutil.to_int(filename,-1) for filename in dl_filename_list]
        dl_filename_int = max(dl_filename_int_list)
        dc_path = os.path.join(dl_path,str(dl_filename_int),'dlcompare')

        dc_filename_list = os.listdir(dc_path)
        dc_filename_int_list = [myutil.to_int(filename[:-5],-1) for filename in dc_filename_list if filename.endswith('.json')]
        dc_filename_int = max(dc_filename_int_list)
        arg_dict['filename'] = os.path.join(dc_path,'{}.json'.format(dc_filename_int))

    data_dict = {}
    with open(arg_dict['filename'],'r') as in_file:
        result_list = json.load(in_file)

    itr_count = 0
    for result in result_list:
        if not 'itr' in result['input']:
            continue
        itr_count = max(itr_count,result['input']['itr']+1)

    win_key_list = [
        {'type':'win','name':'pd','win':'X','style':'ro'},
        {'type':'win','name':'dp','win':'O','style':'rx'},
        {'type':'win','name':'rd','win':'O','style':'go'},
        {'type':'win','name':'dr','win':'X','style':'gx'},
        {'type':'win','name':'rd','win':'X','style':'bo'},
        {'type':'win','name':'dr','win':'O','style':'bx'},
        {'type':'badmove','name':MY_NAME,'style':'k+'},
    ]
    for win_key in win_key_list:
        if win_key['type'] == 'win':
            name = 'win.{}.{}'.format(win_key['name'],win_key['win'])
            data_dict[name] = {}
            data_dict[name]['key'] = win_key
            data_dict[name]['data'] = [0]*itr_count
            for result in result_list:
                if not 'itr' in result['input']:
                    continue
                if result['input']['itr'] >= itr_count:
                    continue
                if not result['input']['name'] == win_key['name']:
                    continue
                data_dict[name]['data'][result['input']['itr']] = result['result']['win_count_dict'][win_key['win']]
        elif win_key['type'] == 'badmove':
            name = 'badmove.{}'.format(win_key['name'])
            data_dict[name] = {}
            data_dict[name]['key'] = win_key
            data_dict[name]['data'] = [0]*itr_count
            for result in result_list:
                if not 'itr' in result['input']:
                    continue
                if result['input']['itr'] >= itr_count:
                    continue
                if result['input']['X']['type'] == win_key['name']:
                    data_dict[name]['data'][result['input']['itr']] += result['result']['bad_move_count_dict']['X']
                if result['input']['O']['type'] == win_key['name']:
                    data_dict[name]['data'][result['input']['itr']] += result['result']['bad_move_count_dict']['O']
#             for itr in range(len(data_dict[name]['data'])):
#                 data_dict[name]['data'][itr] = min(data_dict[name]['data'][itr],1000)
        plt.ylim(0,1000)
        plt.plot(range(itr_count),data_dict[name]['data'],win_key['style'])

    plt.show()
