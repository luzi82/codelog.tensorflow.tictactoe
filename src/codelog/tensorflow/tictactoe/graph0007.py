'''
Created on Jul 12, 2016

@author: luzi82
'''

import json
import matplotlib.pyplot as plt

ITR_COUNT = 1000

if __name__ == '__main__':

    data_dict = {}
    with open('output/dlcompare/1481187874.json','r') as in_file:
        result_list = json.load(in_file)

    win_key_list = [
        {'type':'win','name':'p7','win':'X','style':'ro'},
        {'type':'win','name':'7p','win':'O','style':'rx'},
        {'type':'win','name':'r7','win':'O','style':'go'},
        {'type':'win','name':'7r','win':'X','style':'gx'},
        {'type':'win','name':'r7','win':'X','style':'bo'},
        {'type':'win','name':'7r','win':'O','style':'bx'},
        {'type':'badmove','name':'dl0007','style':'k+'},
    ]
    for win_key in win_key_list:
        if win_key['type'] == 'win':
            name = 'win.{}.{}'.format(win_key['name'],win_key['win'])
            data_dict[name] = {}
            data_dict[name]['key'] = win_key
            data_dict[name]['data'] = [0]*ITR_COUNT
            for result in result_list:
                if not 'itr' in result['input']:
                    continue
                if result['input']['itr'] >= ITR_COUNT:
                    continue
                if not result['input']['name'] == win_key['name']:
                    continue
                data_dict[name]['data'][result['input']['itr']] = result['result']['win_count_dict'][win_key['win']]
        elif win_key['type'] == 'badmove':
            name = 'badmove.{}'.format(win_key['name'])
            data_dict[name] = {}
            data_dict[name]['key'] = win_key
            data_dict[name]['data'] = [0]*ITR_COUNT
            for result in result_list:
                if not 'itr' in result['input']:
                    continue
                if result['input']['itr'] >= ITR_COUNT:
                    continue
                if result['input']['X']['type'] == win_key['name']:
                    data_dict[name]['data'][result['input']['itr']] += result['result']['bad_move_count_dict']['X']
                if result['input']['O']['type'] == win_key['name']:
                    data_dict[name]['data'][result['input']['itr']] += result['result']['bad_move_count_dict']['O']
#             for itr in range(len(data_dict[name]['data'])):
#                 data_dict[name]['data'][itr] = min(data_dict[name]['data'][itr],1000)
        plt.ylim(0,1000)
        plt.plot(range(ITR_COUNT),data_dict[name]['data'],win_key['style'])

    plt.show()
