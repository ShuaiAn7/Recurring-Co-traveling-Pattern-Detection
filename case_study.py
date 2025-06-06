# -*- coding: utf-8 -*-
"""
@author: Shuai An
"""

from RCPD_miner import RCPD_miner_faster
import time
import random
random.seed(2025)
import pandas as pd
import joblib


total_time_start = time.time()
epsilon_logging = {}
epsilon_logging['runtime'] = {}
epsilon_logging['psco_1'] = {}
epsilon_logging['psco_2'] = {}
epsilon_logging['psco_k_refined'] = {}
agents = pd.read_parquet('part_1.parquet')


##########  Get Case Study Results ##################

for i in [0.1, 0.5, 0.9, 99]:
    epsilon_logging['runtime'][str(i)] = []

    pm_faster = RCPD_miner_faster(epsilon_location=i, epsilon_time=i)    
    start_time = time.time()
    process_agents = pm_faster.preprocess_traj(agents)
    end_time = time.time()
    running_time = (end_time-start_time)/60
    print(f"epsilon = {i} data preprocessing time mins: {running_time}")
    
    start_time = time.time()
    psco_faster_1 = pm_faster.mine_PSCO_1(process_agents)
    end_time = time.time()
    running_time = (end_time-start_time)/60
    epsilon_logging['runtime'][str(i)].append(running_time)
    print(f"epsilon= {i} PSCO_1 faster mining time mins: {running_time}")
    epsilon_logging['psco_1'][f"{i}_faster"] = psco_faster_1
     
    start_time = time.time()
    psco_faster_2 = pm_faster.mine_PSCO_2(psco_faster_1)
    end_time = time.time()
    running_time = (end_time-start_time)/60
    epsilon_logging['runtime'][str(i)].append(running_time)
    print(f"epsilon= {i} PSCO_2 faster mining time mins: {running_time}")
    epsilon_logging['psco_2'][f"{i}_faster"] = psco_faster_2
    
    if psco_faster_2 == {}:
        epsilon_logging['runtime'][str(i)].append(-1)
        print(f"epsilon= {i} PSCO_k faster mine_refining time mins: -1")
        epsilon_logging['psco_k_refined'][f"{i}_faster"] = "No PSCO_K Found"
       
    else:    
        start_time = time.time()
        psco_faster_mine_refined = pm_faster.mine_refine_PSCO_k(psco_faster_2)
        end_time = time.time()
        running_time = (end_time-start_time)/60
        epsilon_logging['runtime'][str(i)].append(running_time)
        print(f"epsilon= {i} PSCO_k faster mine_refining time mins: {running_time}")
        epsilon_logging['psco_k_refined'][f"{i}_faster"] = psco_faster_mine_refined
    

total_time_end = time.time()

print("Total running time mins are: ", (total_time_end - total_time_start)/60)

joblib.dump(epsilon_logging, "case_results.joblib")



############## Analyze Case Study Results ######################

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
case_results = joblib.load("case_results.joblib")

########## Use results without noise ################
parco_k = case_results['psco_k_refined']
epsilon99 = parco_k['99_faster']

size_dict = pd.DataFrame([{'key': k, 'num_groups': len(v)} for k, v in epsilon99.items()])

size_dict['size_group'] = size_dict['key'].str.extract(r'_(\d+)', expand=False).astype(int)

plt.figure(figsize=(12, 6))
bars = plt.bar(size_dict['size_group'], size_dict['num_groups'])
plt.xlabel('Recurring Co-traveling Group Size', fontsize=22, fontweight='bold')
plt.ylabel('Number of Groups Found', fontsize=22, fontweight='bold')
plt.xticks(size_dict['size_group'], fontweight='bold')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=20, fontweight='bold')
plt.tick_params(axis='both', labelsize=20)
for label in plt.gca().get_yticklabels():
    label.set_fontweight('bold')
    

plt.tick_params(axis='both', labelsize=20)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


import copy

def mbr_center(mbr):
    y_center = (mbr['bottom_left']['y_meters'] + mbr['top_right']['y_meters']) / 2
    x_center = (mbr['bottom_left']['x_meters'] + mbr['top_right']['x_meters']) / 2
    return (y_center, x_center)

def euclidean_distance(p1, p2):
    dy = p1[0] - p2[0]
    dx = p1[1] - p2[1]
    return (dy**2 + dx**2) ** 0.5

def remove_close_mbrs_and_small_lists(nested_dict, threshold=500):
    new_nested_dict = copy.deepcopy(nested_dict)
    for main_key, subdict in list(new_nested_dict.items()):
        for sub_key, mbr_list in list(subdict.items()):
            cleaned_list = []
            centers = []        
            for time_slot, mbr in mbr_list:
                center = mbr_center(mbr)
                too_close = False
                for existing_center in centers:
                    if euclidean_distance(center, existing_center) < threshold:
                        too_close = True
                        break
                if not too_close:
                    cleaned_list.append((time_slot, mbr))
                    centers.append(center)                  
            if len(cleaned_list) >= 2:
                subdict[sub_key] = cleaned_list
            else:
                del subdict[sub_key]                 
        if not subdict:
            del new_nested_dict[main_key]    
    return new_nested_dict

cleaned_nested_dict = remove_close_mbrs_and_small_lists(epsilon99, threshold=500)


import matplotlib.pyplot as plt


def plot_mbr_subdict(subdict, title='Connected MBR Centers with Time Labels'):
    plt.figure(figsize=(12, 6)) 
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black']
    color_idx = 0
    for sub_key, mbr_list in subdict.items():
        centers = []
        time_slots = []     
        for time_slot, mbr in mbr_list:
            y_center = (mbr['bottom_left']['y_meters'] + mbr['top_right']['y_meters']) / 2
            x_center = (mbr['bottom_left']['x_meters'] + mbr['top_right']['x_meters']) / 2
            centers.append((x_center, y_center))  
            time_slots.append(time_slot)     
        x_coords, y_coords = zip(*centers)
    
        plt.plot(x_coords, y_coords, marker='o', label=f'{sub_key}', color=colors[color_idx % len(colors)], linewidth=3)
        
        for x, y, label in zip(x_coords, y_coords, time_slots):
            plt.text(x +2000, y, label, fontsize=18, ha='right', va='bottom', fontweight='bold')
        
        color_idx += 1  

    plt.xlabel('X (kilometers)', fontsize=22, fontweight='bold')
    plt.ylabel('Y (kilometers', fontsize=22, fontweight='bold')
    plt.tick_params(axis='both', labelsize=18)  
    for label in plt.gca().get_xticklabels():
        label.set_fontweight('bold')
    for label in plt.gca().get_yticklabels():
        label.set_fontweight('bold')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/1000:.0f}'))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}'))
    plt.legend(prop={'size': 16, 'weight': 'bold'})
    plt.grid(True)
    plt.margins(x=0.1, y=0.1)
    plt.tight_layout()
    plt.show()

parco_2 = cleaned_nested_dict['PSCO_2']
parco_2 = {k.replace('PSCO_', 'PARCO_', 1): v for k, v in parco_2.items()}
parco_3 = cleaned_nested_dict['PSCO_3']
parco_3 = {k.replace('PSCO_', 'PARCO_', 1): v for k, v in parco_3.items()}

parco_2_select_keys = ['PARCO_2_59', 'PARCO_2_29']
parco_2_select = {k: parco_2[k] for k in parco_2_select_keys if k in parco_2}
plot_mbr_subdict(parco_2_select, title='Examples of Three-agent co-traveling Routes')

parco_3_select_keys = ['PARCO_3_178', 'PARCO_3_120']
parco_3_select = {k: parco_3[k] for k in parco_3_select_keys if k in parco_3}

plot_mbr_subdict(parco_3_select, title='Examples of Three-agent co-traveling Routes')

total_select_parco = parco_2_select | parco_3_select

key_map = {
    'PARCO_2_29': 'Two-agent group #29',
    'PARCO_2_59': 'Two-agent group #59',
    'PARCO_3_120': 'Three-agent group #120',
    'PARCO_3_178': 'Three-agent group #178'}

total_parco_plot = {key_map[k]: v for k, v in total_select_parco.items()}

name_map = {
    'weekend_0_hour_0': 'Weekday 0-1 am',
    'weekend_0_hour_9': 'Weekday 9-10 am',
    'weekend_0_hour_0': 'Weekday 0-1 am',
    'weekend_0_hour_16': 'Weekday 4-5 pm',
    'weekend_0_hour_7': 'Weekday 7-8 am',
    'weekend_0_hour_18': 'Weekday 6-7 pm',
    'weekend_0_hour_11': 'Weekday 11-12 am',
    'weekend_0_hour_17': 'Weekday 5-6 pm'}

for outer_key, tuple_list in total_parco_plot.items():
    total_parco_plot[outer_key] = [(name_map.get(t[0], t[0]), t[1]) for t in tuple_list]

plot_mbr_subdict(total_parco_plot, title='Examples of Co-traveling Groups')


