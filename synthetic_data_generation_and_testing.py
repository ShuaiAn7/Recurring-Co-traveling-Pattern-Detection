# -*- coding: utf-8 -*-
"""
@author: Shuai An
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
from mbr_functions import mbr_to_mbr_euclidean_distance
from RCPD_miner import RCPD_miner_faster, RCPD_miner_original
import joblib
import time


def generate_valid_center(x_min, x_max, y_min, y_max, circle_radius):
    center_x = np.random.uniform(x_min + circle_radius, x_max - circle_radius)
    center_y = np.random.uniform(y_min + circle_radius, y_max - circle_radius)
    return center_x, center_y

def move_valid_point(original_point, move_distance, x_min, x_max, y_min, y_max, circle_radius):
    max_attempts = 100
    for _ in range(max_attempts):
        move_angle = np.random.uniform(0, 2*np.pi)
        moved_x = original_point[0] + move_distance * np.cos(move_angle)
        moved_y = original_point[1] + move_distance * np.sin(move_angle)

        if (x_min + circle_radius <= moved_x <= x_max - circle_radius) and \
           (y_min + circle_radius <= moved_y <= y_max - circle_radius):
            return moved_x, moved_y
    raise ValueError("Failed to move point")

def create_agents_and_points(x_min, x_max, y_min, y_max,
                              circle_radius, move_distance,
                              n_local_points, n_extra_points, n_extra_agents,
                              skip_agents=False):
    agent_points = []
    agent_centers = []

    if not skip_agents:
        center_x, center_y = generate_valid_center(x_min, x_max, y_min, y_max, circle_radius)
        agent_centers.append((center_x, center_y))

        for _ in range(n_extra_agents):
            moved_x, moved_y = move_valid_point((center_x, center_y), move_distance,
                                                x_min, x_max, y_min, y_max, circle_radius)
            agent_centers.append((moved_x, moved_y))

        for center in agent_centers:
            rand_radii_local = np.sqrt(np.random.uniform(0, 1, n_local_points)) * circle_radius
            rand_angles_local = np.random.uniform(0, 2*np.pi, n_local_points)

            points_local_x = center[0] + rand_radii_local * np.cos(rand_angles_local)
            points_local_y = center[1] + rand_radii_local * np.sin(rand_angles_local)
            points_local = np.column_stack((points_local_x, points_local_y))

            points_extra_x = np.random.uniform(x_min, x_max, size=n_extra_points)
            points_extra_y = np.random.uniform(y_min, y_max, size=n_extra_points)
            points_extra = np.column_stack((points_extra_x, points_extra_y))

            points_combined = np.vstack([points_local, points_extra])
            agent_points.append((center, points_combined))
    else:
        for _ in range(n_extra_agents + 1):
            points_extra_x = np.random.uniform(x_min, x_max, size=n_local_points + n_extra_points)
            points_extra_y = np.random.uniform(y_min, y_max, size=n_local_points + n_extra_points)
            points_extra = np.column_stack((points_extra_x, points_extra_y))
            agent_points.append((None, points_extra))

    return agent_points, agent_centers


##### The generate map function generates the synthetic data, saves the data, and plots the data #########

def generate_map(
    n_quadrants_to_skip=2,
    circle_radius=5,
    move_distance=15,
    n_local_points=20,
    n_extra_points=30,
    n_extra_agents=2,
    study_area=(-100, 100, -100, 100),
    random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 30


    fig, ax = plt.subplots(figsize=(10, 10))
    
 
    ax.set_xlabel("X (meters)", fontsize=30, fontweight='bold')
    ax.set_ylabel("Y (meters)", fontsize=30, fontweight='bold')
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(30)
        label.set_fontweight('bold')
    

    study_x_min, study_x_max, study_y_min, study_y_max = study_area
    ax.set_xlim(study_x_min, study_x_max)
    ax.set_ylim(study_y_min, study_y_max)
    ax.set_aspect('equal')

    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    quadrants = {
        'Q1': (0, study_x_max, 0, study_y_max),
        'Q2': (study_x_min, 0, 0, study_y_max),
        'Q3': (study_x_min, 0, study_y_min, 0),
        'Q4': (0, study_x_max, study_y_min, 0),}

    quadrants_list = list(quadrants.keys())
    if n_quadrants_to_skip > 4:
        raise ValueError("Cannot skip more than 4 quadrants")
    quadrants_to_skip = random.sample(quadrants_list, n_quadrants_to_skip)


    points_records = []
    agent_point_counters = {}
    quadrant_agent_centers = {}
    mbr_info = {}

    for quadrant_label, (x_min, x_max, y_min, y_max) in quadrants.items():
        skip_agents = quadrant_label in quadrants_to_skip

        agent_points, agent_centers = create_agents_and_points(
            x_min, x_max, y_min, y_max,
            circle_radius, move_distance,
            n_local_points, n_extra_points, n_extra_agents,
            skip_agents=skip_agents)
        
        quadrant_agent_centers[quadrant_label] = agent_centers
        

        colors = ['blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'green', 'red', 'black', 'lime']
        markers = ['o', '^', 's', 'd', 'x', '*', 'P', 'h', '+', 'v']

        for agent_idx, (center, points) in enumerate(agent_points, start=1):
            agent_name = f'Agent {agent_idx}'
            color = colors[(agent_idx-1) % len(colors)]
            marker = markers[(agent_idx-1) % len(markers)]

            if center is not None:
                circle = plt.Circle(center, circle_radius, color=color, fill=False, linewidth=2)
                ax.add_artist(circle)

            label = agent_name if quadrant_label == 'Q1' else None

            ax.scatter(points[:, 0], points[:, 1], marker=marker, s=25,
                       color=color, alpha=1.0, label=label)

            if agent_name not in agent_point_counters:
                agent_point_counters[agent_name] = 1

            for (x, y) in points:
                points_records.append({
                    'quadrant': quadrant_label,
                    'agent': agent_name,
                    'point_order': agent_point_counters[agent_name],
                    'x_location': x,
                    'y_location': y
                })
                agent_point_counters[agent_name] += 1

    time_labels = {
        'Q1': 'Time 1',
        'Q2': 'Time 2',
        'Q3': 'Time 3',
        'Q4': 'Time 4'
    }

    for quadrant_label, agent_centers_list in quadrant_agent_centers.items():
        if not agent_centers_list:
            continue  

        centers = np.array(agent_centers_list)

        xmin = np.min(centers[:, 0]) - circle_radius
        xmax = np.max(centers[:, 0]) + circle_radius
        ymin = np.min(centers[:, 1]) - circle_radius
        ymax = np.max(centers[:, 1]) + circle_radius

        width = xmax - xmin
        height = ymax - ymin

        rect = plt.Rectangle((xmin, ymin), width, height,
                              linewidth=3, edgecolor='black', facecolor='none', linestyle='-')
        ax.add_patch(rect)

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        mbr_info[quadrant_label] = (center_x, center_y, width, height)

        label = time_labels.get(quadrant_label, '')
        label_offset_x = -5
        label_offset_y = 3
        label_x = xmin + label_offset_x
        label_y = ymax + label_offset_y
        
        ax.text(label_x, label_y, label, fontsize=30, fontweight='bold',
                ha='left', va='bottom', color='black')



    ordered_quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    mbr_centers_ordered = [mbr_info[q] for q in ordered_quadrants if q in mbr_info]

    if len(mbr_centers_ordered) >= 2:
        for i in range(len(mbr_centers_ordered) - 1):
            start_cx, start_cy, start_w, start_h = mbr_centers_ordered[i]
            end_cx, end_cy, end_w, end_h = mbr_centers_ordered[i + 1]

            start = np.array([start_cx, start_cy])
            end = np.array([end_cx, end_cy])

            direction = end - start
            length = np.linalg.norm(direction)
            if length != 0:
                direction /= length

                offset_start = np.array([start_w/2, start_h/2]) * np.abs(direction)
                offset_end = np.array([end_w/2, end_h/2]) * np.abs(direction)

                start_adj = start + direction * np.max(offset_start)
                end_adj = end - direction * np.max(offset_end)

                ax.annotate('',
                            xy=end_adj, xytext=start_adj,
                            arrowprops=dict(arrowstyle='->', color='black', lw=4, mutation_scale=20))

    ax.grid(True)

    leg = ax.legend(loc='lower right', frameon=True)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(1)
    
    for handle in leg.legend_handles:
        handle.set_sizes([100])
    for text in leg.get_texts():
        text.set_fontsize(30)  
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.show()

    points_df = pd.DataFrame(points_records)
    
    mbr_list = []
    for quadrant_label, (center_x, center_y, width, height) in mbr_info.items():
        if quadrant_label == 'Q1':
            time_window = 'weekend_0_hour_9'
        elif quadrant_label == 'Q2':
            time_window = 'weekend_0_hour_17'
        elif quadrant_label == 'Q3':
            time_window = 'weekend_1_hour_12'
        elif quadrant_label == 'Q4':
            time_window = 'weekend_1_hour_20'
        
        half_width = width / 2
        half_height = height / 2
    
        mbr_corners = {
            'top_left': {'x_meters': center_x - half_width, 'y_meters': center_y + half_height},
            'top_right': {'x_meters': center_x + half_width, 'y_meters': center_y + half_height},
            'bottom_left': {'x_meters': center_x - half_width, 'y_meters': center_y - half_height},
            'bottom_right': {'x_meters': center_x + half_width, 'y_meters': center_y - half_height}
        }
    
        mbr_list.append((time_window, mbr_corners))

    return {'data': points_df, 'ground_truth': mbr_list, 'n_agents': n_extra_agents +1}





def mbrA_contained_in_mbrB(mbr_a, mbr_b):

    return (
        mbr_a["top_left"]['y_meters'] <= mbr_b["top_left"]['y_meters']  
        and mbr_a["top_left"]['x_meters'] >= mbr_b["top_left"]['x_meters']  
        and mbr_a["bottom_right"]['y_meters'] >= mbr_b["bottom_right"]['y_meters']  
        and mbr_a["bottom_right"]['x_meters'] <= mbr_b["bottom_right"]['x_meters'])




def find_contained_groups(PSCO_small, PSCO_large):

    large_groups_dict = {
        frozenset(group_b): dict(location_b) for group_b, location_b in PSCO_large.items()
    }

    contained_groups = []

    for group_a, location_a in PSCO_small.items():
        group_a_set = frozenset(group_a)
        a_dict = dict(location_a)  

        for group_b_set, b_dict in large_groups_dict.items():
            if not group_a_set.issubset(group_b_set):
                continue

            if not set(a_dict.keys()).issubset(set(b_dict.keys())):
                continue

            if all(mbrA_contained_in_mbrB(a_dict[tw], b_dict[tw]) for tw in a_dict):
                contained_groups.append(group_a)
                break  
    return contained_groups




def same_results(ground_truth, mine_results, distance_threshold=5):

    if len(mine_results) != 1:
        print("Different size co-traveling group found")
        return False
    mine_key = next(iter(mine_results))
    group_dict = mine_results[mine_key]
    if len(group_dict) != 1:
        print("Multiple co-traveling group found")
        return False
    n_agents_g = ground_truth['n_agents']
    n_agents_m = int(mine_key.split('_')[-1]) 
    if n_agents_m != n_agents_g:
        print("Number of agents does not match")
        return False
    
    route_g = ground_truth['ground_truth']
    route_m = next(iter(group_dict.values()))

    
    route_g_dict = dict(route_g)
    route_m_dict = dict(route_m)
    if len(route_g) != len(route_m):
        print("Number of time windows does not match")
        return False
    same_keys = set(route_g_dict.keys()) == set(route_m_dict.keys())
    if not same_keys:
        print("Time windows do not match")
        return False
    for i in route_m_dict.keys():
        mbr_g = route_g_dict[i]
        mbr_m = route_m_dict[i]
        distance = mbr_to_mbr_euclidean_distance(mbr_g, mbr_m)
        if distance >= distance_threshold:
            print("MBR distance too far")
            return False
    return True
    

def testing(n_agents, n_time_window, diff_privacy, epsilon_location, epsilon_time, sensitivity_meters, sensitivity_seconds, apply_time_noise):
    TP = 0
    FP = 0
    TN = 0 
    FN = 0
    for i in n_agents:
        for j in n_time_window:
            ground_truth = generate_map(n_extra_agents=i-1, n_quadrants_to_skip=4-j)
            data_syn = ground_truth['data']
            data_syn = data_syn.sort_values(by=['agent', 'point_order'])
            data_syn = data_syn.rename(columns={'agent': 'caid', 'point_order': 'utc_timestamp', 'x_location': 'x_meters', 'y_location': 'y_meters'})
            data_syn = data_syn.drop(columns=['quadrant'])
            pm = RCPD_miner_original(threshold_count=2, diff_privacy=diff_privacy, epsilon_location=epsilon_location, epsilon_time=epsilon_time, sensitivity_meters=sensitivity_meters, sensitivity_seconds=sensitivity_seconds, apply_time_noise=apply_time_noise, dbscan_eps=3, threshold_mbr_distance = 20)
            data_syn_processed = pm.preprocess_traj_synthetic(data_syn, 50)
            psco_fast_1 = pm.mine_PSCO_1(data_syn_processed)
            if j <= 1:          
                if len(psco_fast_1) <= 1:
                    TN += 1
                else:
                    psco_fast_2 = pm.mine_PSCO_2(psco_fast_1)
                    if psco_fast_2 == {}:
                        TN += 1
                    else:
                        FP += 1
            else:
                if len(psco_fast_1) <= 1:
                    FN += 1
                else:
                    psco_fast_2 = pm.mine_PSCO_2(psco_fast_1)
                    if psco_fast_2 == {}:
                        FN += 1 
                    else:
                        mine_results = pm.mine_refine_PSCO_k(psco_fast_2)
                        if same_results(ground_truth, mine_results, distance_threshold=5):
                            TP += 1
                        else:
      
                            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if TP == 0:
        precision = 0  
        recall = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
 
    print(f"ACCURACY = {accuracy}, PRECISION = {precision}, RECALL = {recall}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall}




n_agents = [2, 3, 4]
n_time_window = [0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100


epsilon_sensitivity = {}  
runtime = {}   
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 99]:
    start_time = time.time()         
    result_i = testing(n_agents, n_time_window, diff_privacy = True, epsilon_location = i, epsilon_time = i, sensitivity_meters = 1, sensitivity_seconds = 1, apply_time_noise = True)
    end_time = time.time()
    rt = (end_time - start_time) / 60
    print(f"running time is: {rt}")
    runtime[f'{i}'] = rt
    epsilon_sensitivity[f'{i}'] = result_i      

          
    
joblib.dump(epsilon_sensitivity, "epsilon_testing_results.joblib")


epsilon_sensitivity = joblib.load('epsilon_testing_results.joblib')

epsilon_sensitivity['no noise'] = epsilon_sensitivity.pop('99')

eps_level = ['0.1', '0.2', '0.3', '0.4', '0.5', 'no noise']  
metrics = ['Precision', 'Accuracy', 'Recall']  


values = [[epsilon_sensitivity[e_l][met] for e_l in eps_level] for met in metrics]

x = np.arange(len(eps_level))
bar_width = 0.2
fig, ax = plt.subplots(figsize=(8, 5))
for i, (cat, vals) in enumerate(zip(metrics, values)):
    ax.bar(x + i * bar_width, vals, bar_width, label=cat)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(eps_level)
ax.set_ylabel("Score", fontsize=16)
ax.set_xlabel("Noise Level ($\epsilon$)", fontsize=16)
ax.legend(fontsize = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()


runtime_logged = [8.703044565518697, 8.870938766002656, 8.84243845542272, 8.761158812046052, 8.686522404352823, 7.644061172008515]

x = np.arange(len(eps_level))
bar_width = 0.2

fig, ax1 = plt.subplots(figsize=(10, 5))

for i, (cat, vals) in enumerate(zip(metrics, values)):
    ax1.bar(x + i * bar_width, vals, bar_width, label=cat)
    
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(eps_level)
ax1.set_ylabel("Score", fontsize=18, fontweight='bold')
ax1.set_xlabel("Noise Level ($\epsilon$)", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='blue')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(x + bar_width * len(metrics) / 2, runtime_logged, color='red', marker='^', label='Runtime', linewidth = 2, markersize=10)
ax2.set_ylabel("Runtime (Minutes)", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red')
for label in ax2.get_yticklabels():
    label.set_fontsize(16)
    label.set_fontweight('bold')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.1, 0.4), prop={'size': 16, 'weight': 'bold'})
plt.tight_layout()
plt.show()

