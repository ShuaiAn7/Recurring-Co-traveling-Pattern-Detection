# -*- coding: utf-8 -*-
"""
@author: Shuai An
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np



def convert_runtime_dict(time_dict, key_type, var_name):
    runtime_lst = []
    for key, value in time_dict.items():
        orig_time = value['data_process_' + key + '_original'] + value['psco1_' + key + '_original'] + value['psco2_' + key + '_original'] + value['pscok_' + key + '_original']
        faster_time = value['data_process_' + key + '_faster'] + value['psco1_' + key + '_faster'] + value['psco2_' + key + '_faster'] + value['pscok_' + key + '_faster']
        if key_type == 'int':
            runtime_lst.append((int(key), orig_time, faster_time))
        elif key_type == 'float':
            runtime_lst.append((float(key), orig_time, faster_time))
    runtime_df = pd.DataFrame(runtime_lst, columns = [var_name, 'total_original_time', 'total_faster_time'])
    return runtime_df



################################### Number of Agents #############################
from matplotlib.ticker import FuncFormatter
n_agents_logging = joblib.load("n_agents.joblib")


runtime = n_agents_logging['runtime']
runtime_df = convert_runtime_dict(runtime, 'int', 'n_agents')
runtime_df['n_agents'] = runtime_df['n_agents'] / 1000


selected_n_agents = [3, 6, 9, 12, 15]

runtime_df = runtime_df[runtime_df['n_agents'].isin(selected_n_agents)]


plt.figure(figsize=(7, 7))  
plt.plot(runtime_df["n_agents"], np.log(runtime_df["total_original_time"]), label="RCPD", marker='o', color = 'blue', linestyle = '--', markersize=12,linewidth=3,)
plt.plot(runtime_df["n_agents"], np.log(runtime_df["total_faster_time"]), label="RCPD-Adv", marker='^', color = 'red', linestyle = '-', markersize=12, linewidth=3,)
plt.xlabel("Number of Agents (Thousands)", fontsize = 23, fontweight='bold')
plt.xticks(selected_n_agents, fontweight='bold', fontsize=22)
plt.yticks(fontweight='bold', fontsize=22)
plt.ylabel("Runtime Log(minutes)", fontsize = 23, fontweight='bold')
plt.legend(fontsize = 20, prop={'weight': 'bold'})
plt.grid(True)
plt.show()


psco_k = n_agents_logging['psco_k_refined']

for key, value in psco_k.items():
    if isinstance(value, str):  
        psco_k[key] = {}  
        
rows = []
for key, value_dict in psco_k.items():
    num_elements = len(value_dict) + 1  
    total_sub_elements = sum(len(sub_dict) for sub_dict in value_dict.values())
    rows.append([key, num_elements, total_sub_elements])

psco_k_df = pd.DataFrame(rows, columns=["n_agents", "Max Group", "Total PSCOs"])


psco_k_df[['n_agents', 'version']] = psco_k_df['n_agents'].str.split('_', expand=True)
psco_k_df['n_agents'] = psco_k_df['n_agents'].astype(int)
psco_k_df['n_agents'] = psco_k_df['n_agents'] / 1000
psco_k_df = psco_k_df[psco_k_df['n_agents'].isin(selected_n_agents)]
psco_k_df_pivot = psco_k_df.pivot(index='n_agents', columns='version', values=['Max Group', 'Total PSCOs'])
psco_k_df_pivot.columns = [f"{col[0]}_{col[1]}" for col in psco_k_df_pivot.columns]
psco_k_df_pivot = psco_k_df_pivot.reset_index()


x = psco_k_df_pivot['n_agents'].astype(int)
x_indexes = np.arange(len(x))
bar_width = 0.15
group_gap = 0.05 

fig, ax1 = plt.subplots(figsize=(7, 6))

ax1.bar(x_indexes - bar_width - group_gap, psco_k_df_pivot['Total PSCOs_faster'], width=bar_width, label='Total Groups (RCPD-Adv)', color='grey')
ax1.bar(x_indexes-group_gap, psco_k_df_pivot['Total PSCOs_original'], width=bar_width, label='Total Groups (RCPD)', color='brown')
ax1.set_ylabel('Total Groups (Hundreds)', fontsize=20, fontweight='bold')
ax1.set_xlabel('Number of Agents (Thousands)', fontsize=20, fontweight='bold')
ax1.set_xticks(x_indexes)
ax1.set_xticklabels(x, fontsize=19, fontweight='bold')
ax1.tick_params(axis='x', pad=1, labelsize=19)  
ax1.tick_params(axis='y', pad=1, labelsize=19)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/100:.0f}'))
ax2 = ax1.twinx()
ax2.bar(x_indexes + bar_width, psco_k_df_pivot['Max Group_faster'], width=bar_width, label='Largest Group\n(RCPD-Adv)', color='blue')
ax2.bar(x_indexes + 2 * bar_width, psco_k_df_pivot['Max Group_original'], width=bar_width, label='Largest Group\n(RCPD)', color='red')
ax2.set_ylabel('Largest Group Size', fontsize=20, fontweight='bold')
ax2.tick_params(axis='y', labelsize=19)

for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=1, bbox_to_anchor=(0.415, 0.97), prop={'size': 15, 'weight': 'bold'}, framealpha=0)
plt.tight_layout()
plt.show()



################################### Threshold Count #############################

n_counts_3_logging = joblib.load("n_counts_logging.joblib")
runtime = n_counts_3_logging['runtime']
runtime_df = convert_runtime_dict(runtime, 'int', 'n_counts_3')
runtime_df = runtime_df[runtime_df['n_counts_3'] <= 15]

plt.figure(figsize=(7, 7))  
plt.plot(runtime_df['n_counts_3'], np.log(runtime_df["total_original_time"]), label="RCPD", marker='o', color = 'blue', linestyle = '--', markersize=12,linewidth=3,)
plt.plot(runtime_df['n_counts_3'], np.log(runtime_df["total_faster_time"]), label="RCPD-Adv", marker='^', color = 'red', linestyle = '-', markersize=12, linewidth=3,)
plt.xlabel("Minimum Time Windows", fontsize = 23, fontweight='bold')
plt.xticks(selected_n_agents, fontweight='bold', fontsize=22)
plt.yticks(fontweight='bold', fontsize=22)
plt.ylabel("Runtime Log(minutes)", fontsize = 23, fontweight='bold')
plt.legend(fontsize = 20, prop={'weight': 'bold'})
plt.grid(True)
plt.show()


psco_k = n_counts_3_logging['psco_k_refined']
for key, value in psco_k.items():
    if isinstance(value, str):  
        psco_k[key] = {}      
rows = []
for key, value_dict in psco_k.items():
    num_elements = len(value_dict) + 1  
    total_sub_elements = sum(len(sub_dict) for sub_dict in value_dict.values())
    rows.append([key, num_elements, total_sub_elements])

psco_k_df = pd.DataFrame(rows, columns=["n_counts_3", "Max Group", "Total PSCOs"])
psco_k_df[['count', 'version']] = psco_k_df['n_counts_3'].str.split('_', expand=True)
psco_k_df['count'] = psco_k_df['count'].astype(int)
psco_k_df_pivot = psco_k_df.pivot(index='count', columns='version', values=['Max Group', 'Total PSCOs'])
psco_k_df_pivot.columns = [f"{col[0]}_{col[1]}" for col in psco_k_df_pivot.columns]
psco_k_df_pivot = psco_k_df_pivot.reset_index()

psco_k_df_pivot = psco_k_df_pivot[psco_k_df_pivot['count']<=15]


x = psco_k_df_pivot["count"].astype(int)
x_indexes = np.arange(len(x))
bar_width = 0.15
group_gap = 0.05 

fig, ax1 = plt.subplots(figsize=(7, 6))

ax1.bar(x_indexes - bar_width - group_gap, psco_k_df_pivot['Total PSCOs_faster'], width=bar_width, label='Total Groups (RCPD-Adv)', color='grey')
ax1.bar(x_indexes-group_gap, psco_k_df_pivot['Total PSCOs_original'], width=bar_width, label='Total Groups (RCPD)', color='brown')
ax1.set_ylabel('Total Groups (Hundreds)', fontsize=20, fontweight='bold')
ax1.set_xlabel('Minimum Time Windows', fontsize=20, fontweight='bold')
ax1.set_xticks(x_indexes)
ax1.set_xticklabels(x, fontsize=19, fontweight='bold')
ax1.tick_params(axis='x', pad=1, labelsize=19)  
ax1.tick_params(axis='y', pad=1, labelsize=19)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/100:.0f}'))
ax2 = ax1.twinx()
ax2.bar(x_indexes + bar_width, psco_k_df_pivot['Max Group_faster'], width=bar_width, label='Largest Group (RCPD-Adv)', color='blue')
ax2.bar(x_indexes + 2 * bar_width, psco_k_df_pivot['Max Group_original'], width=bar_width, label='Largest Group (RCPD)', color='red')
ax2.set_ylabel('Largest Group Size', fontsize=20, fontweight='bold')
ax2.tick_params(axis='y', labelsize=19)

for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=1, bbox_to_anchor=(0.55, 0.95), prop={'size': 15, 'weight': 'bold'}, framealpha=0.5)
plt.tight_layout()
plt.show()




################################### Largest Cluster Min Points #############################

cluster_logging = joblib.load("largest_cluster_min_points.joblib")
runtime = cluster_logging['runtime']

runtime_df = convert_runtime_dict(runtime, 'int', 'Largest Cluster Minimum Points')


plt.figure(figsize=(7, 7))  
plt.plot(runtime_df['Largest Cluster Minimum Points'], np.log(runtime_df["total_original_time"]), label="RCPD", marker='o', color = 'blue', linestyle = '--', markersize=12,linewidth=3,)
plt.plot(runtime_df['Largest Cluster Minimum Points'], np.log(runtime_df["total_faster_time"]), label="RCPD-Adv", marker='^', color = 'red', linestyle = '-', markersize=12, linewidth=3,)
plt.xlabel("Largest Cluster Min Points", fontsize = 23, fontweight='bold')
plt.xticks(fontweight='bold', fontsize=22)
plt.yticks(fontweight='bold', fontsize=22)
plt.ylabel("Runtime Log(minutes)", fontsize = 23, fontweight='bold')
plt.legend(fontsize = 20, prop={'weight': 'bold'})
plt.grid(True)
plt.show()



psco_k = cluster_logging['psco_k_refined']

for key, value in psco_k.items():
    if isinstance(value, str):  
        psco_k[key] = {}  
        
rows = []
for key, value_dict in psco_k.items():
    num_elements = len(value_dict) + 1  
    total_sub_elements = sum(len(sub_dict) for sub_dict in value_dict.values())
    rows.append([key, num_elements, total_sub_elements])

psco_k_df = pd.DataFrame(rows, columns=["Largest Cluster Minimum Points", "Max Group", "Total PSCOs"])

psco_k_df[['count', 'version']] = psco_k_df['Largest Cluster Minimum Points'].str.split('_', expand=True)
psco_k_df['count'] = psco_k_df['count'].astype(int)

psco_k_df_pivot = psco_k_df.pivot(index='count', columns='version', values=['Max Group', 'Total PSCOs'])

psco_k_df_pivot.columns = [f"{col[0]}_{col[1]}" for col in psco_k_df_pivot.columns]
psco_k_df_pivot = psco_k_df_pivot.reset_index()


x = psco_k_df_pivot["count"].astype(int)
x_indexes = np.arange(len(x))
bar_width = 0.15
group_gap = 0.05 

fig, ax1 = plt.subplots(figsize=(7, 6))
ax1.bar(x_indexes - bar_width - group_gap, psco_k_df_pivot['Total PSCOs_faster'], width=bar_width, label='Total Groups (RCPD-Adv)', color='grey')
ax1.bar(x_indexes-group_gap, psco_k_df_pivot['Total PSCOs_original'], width=bar_width, label='Total Groups (RCPD)', color='brown')
ax1.set_ylabel('Total Groups (Hundreds)', fontsize=20, fontweight='bold')
ax1.set_xlabel('Largest Cluster Min Points', fontsize=20, fontweight='bold')
ax1.set_xticks(x_indexes)
ax1.set_xticklabels(x, fontsize=19, fontweight='bold')
ax1.tick_params(axis='x', pad=1, labelsize=19)  
ax1.tick_params(axis='y', pad=1, labelsize=19)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/100:.0f}'))

ax2 = ax1.twinx()
ax2.bar(x_indexes + bar_width, psco_k_df_pivot['Max Group_faster'], width=bar_width, label='Largest Group (RCPD-Adv)', color='blue')
ax2.bar(x_indexes + 2 * bar_width, psco_k_df_pivot['Max Group_original'], width=bar_width, label='Largest Group (RCPD)', color='red')
ax2.set_ylabel('Largest Group Size', fontsize=20, fontweight='bold')
ax2.tick_params(axis='y', labelsize=19)

for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=1, bbox_to_anchor=(0.55, 0.95), prop={'size': 15, 'weight': 'bold'}, framealpha=0.5)
plt.tight_layout()
plt.show()



