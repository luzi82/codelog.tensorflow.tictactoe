'''
Created on Jul 12, 2016

@author: luzi82
'''

import json
import matplotlib.pyplot as plt

ITR_COUNT = 600

if __name__ == '__main__':
    data = {
        'p0_lose':[0]*ITR_COUNT,
        '0p_lose':[0]*ITR_COUNT,
        'p1_lose':[0]*ITR_COUNT,
        '1p_lose':[0]*ITR_COUNT,
        'r0_lose':[0]*ITR_COUNT,
        '0r_lose':[0]*ITR_COUNT,
        'r1_lose':[0]*ITR_COUNT,
        '1r_lose':[0]*ITR_COUNT,
        'r0_win':[0]*ITR_COUNT,
        '0r_win':[0]*ITR_COUNT,
        'r1_win':[0]*ITR_COUNT,
        '1r_win':[0]*ITR_COUNT,
        '0_badmove':[0]*ITR_COUNT,
        '1_badmove':[0]*ITR_COUNT,
    }
    with open('output/dlcompare/1468141993.json','r') as in_file:
        result_list = json.load(in_file)
    for result in result_list:
        if result['input']['name'] == 'p0' or result['input']['name'] == 'p1':
            data['{}_lose'.format(result['input']['name'])][result['input']['itr']] = result['result']['win_count_dict']['O']

    plt.plot(range(ITR_COUNT),data['p0_lose'],'r^')
#     plt.plot(range(ITR_COUNT),range(ITR_COUNT),'r^')
    plt.show()
