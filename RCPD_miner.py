# -*- coding: utf-8 -*-
"""
@author: Shuai An
"""

import random
random.seed(2025)
from itertools import combinations
import pandas as pd
import numpy as np
from rtree import index

from mbr_functions import calculate_mbr, calculate_mbr_area, mbr_to_mbr_euclidean_distance, mbr_dict_to_tuple, cloak 

from sklearn.cluster import DBSCAN
from pyproj import Transformer


############################ common functions  #######################################

def relative_coordinates(agents):

    reference_idx = np.random.choice(agents.index)
    random_lat = agents.loc[reference_idx, 'latitude']
    random_lon = agents.loc[reference_idx, 'longitude']
    
    proj_string = f"+proj=aeqd +lat_0={random_lat} +lon_0={random_lon} +x_0=0 +y_0=0 +units=m +datum=WGS84"

    lonlat_to_xy = Transformer.from_crs("epsg:4326", proj_string, always_xy=True)
    
    x_meters, y_meters = lonlat_to_xy.transform(agents['longitude'].values, agents['latitude'].values)
    
    agents['x_meters'] = x_meters
    agents['y_meters'] = y_meters   
    agents = agents.drop(columns=['latitude', 'longitude'])
    return agents



def planar_laplace_batch_meters(x_meters, y_meters, epsilon, sensitivity_meters):

    size = len(y_meters)

    theta = np.random.uniform(0, 2 * np.pi, size=size)
    r = np.random.exponential(scale=sensitivity_meters / epsilon, size=size)  

    delta_y = r * np.cos(theta)
    delta_x = r * np.sin(theta)

    noisy_y = y_meters + delta_y
    noisy_x = x_meters + delta_x

    return noisy_x, noisy_y



def add_laplace_noise_timestamp_batch(timestamps, epsilon, sensitivity_seconds):
 
    scale = sensitivity_seconds / epsilon
    noise = np.random.laplace(0, scale, size=len(timestamps))
    noisy_timestamps = timestamps + noise
    return noisy_timestamps


def noise_trajectory_batch_meters(df, epsilon_location=0.5, epsilon_time=0.5,
                                   sensitivity_meters=10, sensitivity_seconds=300,
                                   apply_time_noise=True):

    if epsilon_location == 99 and epsilon_time == 99:
        sensitivity_meters = 0 
        sensitivity_seconds = 0 

    noisy_x_meters, noisy_y_meters = planar_laplace_batch_meters(
        df['x_meters'].values,
        df['y_meters'].values,
        epsilon_location,
        sensitivity_meters
    )


    if apply_time_noise:
        noisy_timestamps = add_laplace_noise_timestamp_batch(
            df['utc_timestamp'].values,
            epsilon_time,
            sensitivity_seconds
        )
    else:
        noisy_timestamps = df['utc_timestamp'].values


    noisy_df = pd.DataFrame({
        'caid': df['caid'].values,
        'y_meters': noisy_y_meters,
        'x_meters': noisy_x_meters,
        'utc_timestamp': noisy_timestamps
    })

    return noisy_df


def add_isweekend_hour(df, step):
    df = df.copy()
    conditions = [
        (df['utc_timestamp'] >= 1) & (df['utc_timestamp'] <= step),
        (df['utc_timestamp'] >= step + 1) & (df['utc_timestamp'] <= 2 * step),
        (df['utc_timestamp'] >= 2 * step + 1) & (df['utc_timestamp'] <= 3 * step),
        (df['utc_timestamp'] >= 3 * step + 1) & (df['utc_timestamp'] <= 4 * step)]    
    isweekend_values = [0, 0, 1, 1]
    hour_values = [9, 17, 12, 20]
    df['IsWeekend'] = np.select(conditions, isweekend_values, default=np.nan)
    df['hour'] = np.select(conditions, hour_values, default=np.nan)
    df = df.dropna(subset=['IsWeekend', 'hour']).reset_index(drop=True)
    df['IsWeekend'] = df['IsWeekend'].astype(int)
    df['hour'] = df['hour'].astype(int)
    return df

def most_visited_cluster(df, eps_m=20, min_samples=3, largest_cluster_min_points=5):
    if df.empty:
        return df
    
    X = df[['x_meters', 'y_meters']].values
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    clusters = db.fit_predict(X)

    df["cluster"] = clusters
  
    df_no_noise = df[df["cluster"] != -1].reset_index(drop=True)
    
    if df_no_noise.empty:
        return df_no_noise
    cluster_counts = df_no_noise.groupby("cluster").size().reset_index(name="count")
    largest_cluster_id = cluster_counts.loc[cluster_counts["count"].idxmax(), "cluster"]
    df_largest_cluster = df_no_noise[df_no_noise["cluster"] == largest_cluster_id].reset_index(drop=True)
    if len(df_largest_cluster) <= largest_cluster_min_points:
        return pd.DataFrame()
    df_largest_cluster.drop(columns=['cluster'], inplace=True)
    return df_largest_cluster




def transform_PSCO(PSCO_k_groups, time_order):
    '''
    transform the dictionary from having the agents as keys to use time_windows
    PSCO_k_groups: a dictionary of colocation groups of all time_windowas keys
    '''
    transformed_PSCO_k = {}
    for agent_id, tuples in PSCO_k_groups.items():
        for time_slot, location_dict in tuples:
            if time_slot not in transformed_PSCO_k:
                transformed_PSCO_k[time_slot] = {}
            transformed_PSCO_k[time_slot][agent_id] = location_dict
    transformed_PSCO_k_sorted = {k: transformed_PSCO_k[k] for k in sorted(transformed_PSCO_k.keys(), key=lambda x: time_order.index(x))}
    return transformed_PSCO_k_sorted        
        

def prune_candidates(candidates, PSCO_k_groups):
    return candidates



def merged_mbr(mbr1, mbr2):
    y_meters1 = [corner["y_meters"] for corner in mbr1.values()]
    x_meters1 = [corner["x_meters"] for corner in mbr1.values()]
    
    y_meters2 = [corner["y_meters"] for corner in mbr2.values()]
    x_meters2 = [corner["x_meters"] for corner in mbr2.values()]
    

    min_y_meters = min(min(y_meters1), min(y_meters2))
    max_y_meters = max(max(y_meters1), max(y_meters2))
    min_x_meters = min(min(x_meters1), min(x_meters2))
    max_x_meters = max(max(x_meters1), max(x_meters2))
    
    enclosing_mbr = {
        "bottom_left": {"y_meters": min_y_meters, "x_meters": min_x_meters},
        "bottom_right": {"y_meters": min_y_meters, "x_meters": max_x_meters},
        "top_left": {"y_meters": max_y_meters, "x_meters": min_x_meters},
        "top_right": {"y_meters": max_y_meters, "x_meters": max_x_meters}}
    return enclosing_mbr


def check_colocation_area(candidates, PSCO_k_groups, max_area):
    '''
    PSCO_k_groups: a dictionary of colocation groups of a given time_window
    '''
    for candidate, (group1, group2) in list(candidates.items()):
        mbr1 = PSCO_k_groups[group1]
        mbr2 = PSCO_k_groups[group2]
        enclosing_mbr = merged_mbr(mbr1, mbr2)
        enclosing_mbr_area = calculate_mbr_area(enclosing_mbr)
        if enclosing_mbr_area > max_area:
            del candidates[candidate] 
        else:
            candidates[candidate] = enclosing_mbr 
    return candidates


def mbrA_contained_in_mbrB(mbr_a, mbr_b):

    return (
        mbr_a["top_left"]['y_meters'] <= mbr_b["top_left"]['y_meters']  
        and mbr_a["top_left"]['x_meters'] >= mbr_b["top_left"]['x_meters']  
        and mbr_a["bottom_right"]['y_meters'] >= mbr_b["bottom_right"]['y_meters']  
        and mbr_a["bottom_right"]['x_meters'] <= mbr_b["bottom_right"]['x_meters'])


def two_agent_colocations(time_window, transformed_PSCO_1, PSCO_1_ids_grid, threshold_mbr_distance, max_area):  

    weekend_hour = transformed_PSCO_1[time_window]
    weekend_hour_nonempty = {k: v for k, v in weekend_hour.items() if v != {}}
    
    ids = sorted(PSCO_1_ids_grid)    
    combination_length = 2
    combinations_list = list(combinations(ids, combination_length))
    
    shortest_dist = []
    for i in combinations_list:
        id1 = i[0]
        id2 = i[1]
        mbr1 = weekend_hour_nonempty[id1]
        mbr2 = weekend_hour_nonempty[id2]
        short = mbr_to_mbr_euclidean_distance(mbr1, mbr2)
        shortest_dist.append(short)
    
    weekend_hour_dist = pd.DataFrame({'group' : combinations_list, 'distance' : shortest_dist})
    
    weekend_hour_g2 = weekend_hour_dist[weekend_hour_dist['distance'] <= threshold_mbr_distance].copy()
    
    if weekend_hour_g2.empty:
        return pd.DataFrame() 

    weekend_hour_g2[['id1', 'id2']] = pd.DataFrame(weekend_hour_g2['group'].tolist(), index=weekend_hour_g2.index)

    weekend_hour_g2['mbr1'] = weekend_hour_g2['id1'].map(weekend_hour_nonempty)
    weekend_hour_g2['mbr2'] = weekend_hour_g2['id2'].map(weekend_hour_nonempty)

    weekend_hour_g2['grouped_points_mbr'] = weekend_hour_g2.apply(lambda row: merged_mbr(row['mbr1'], row['mbr2']),axis=1)

    weekend_hour_g2['grouped_points_mbr_area'] = weekend_hour_g2['grouped_points_mbr'].apply(lambda x: calculate_mbr_area(x))

    weekend_hour_g2_area = weekend_hour_g2[weekend_hour_g2['grouped_points_mbr_area'] <= max_area]
    weekend_hour_g2_area = weekend_hour_g2_area[['group', 'grouped_points_mbr']]
    return weekend_hour_g2_area



############################# faster method ##########################################################
######################################################################################################


class RCPD_miner_faster:
    def __init__(self, K = 2, remove_id = "Yes", diff_privacy = True, epsilon_location=0.5, epsilon_time=0.5, sensitivity_meters=5, sensitivity_seconds=100, apply_time_noise=True, threshold_mbr_distance = 20, max_area = 12100, min_area = 100, x_replace_range = (10, 20), y_replace_range = (10, 20), threshold_count = 3, time_order = [f"weekend_{i}_hour_{j}" for i in range(2) for j in range(24)], dbscan_eps = 20, dbscan_min_samples = 3, largest_cluster_min_points = 5, x_bins = 50, y_bins = 50):
        
        
        '''
        K: minimum number of agents in a co_traveling group
        remove_id: the output contains none agent_ids
        diff_privacy (True/False): whether to add noise to trajectories 
        epsilon_location: location noise level, the smaller, the higher 
        epsilon_time: timestamp noise level, the smaller, the higher 
        sensitivity_meters (m): How much a reported location (latitude/longitude or x/y meters) could change
if one person (or one record) is added, removed, or moved slightly 
        sensitivity_seconds (s): How much a single change (adding/removing/changing one timestamp)
could affect the output of your function. 
        apply_time_noise (True/False): whether to add noise to timestamp
        
        
        threshold_mbr_distance (m): the threshold distance between two mbrs that decide whether two mbrs are colocated.
        
        max_area (m^2): the threshold of mbr area that decides whether a group mbr is small enough to be meaningful.
        min_area (m^2): the threshold of mbr area that decides whether the mbr is too small that needs to be cloaked.
        x_replace_range (m): replace the old mbr with a new mbr with an x-length randomly picked in this range and the original mbr center. 
        y_replace_range (m): replace the old mbr with a new mbr with a y-length randomly picked in this range and the original mbr center.
        
        threshold_count: the smallest number of time windows that the qualified groups colocated.
        time_order: the order of the time windows
        
        
        dbscan_eps (m): the maximum distance between two samples for one to be considered as in the neighborhood of the other.
        dbscan_min_samples: the number of samples in a neighborhood for a point to be considered as a core point.
        largest_cluster_min_points: the minimum number of points a most-visited location must have to be kept.
        
        study_area_mbr: specify the study area in y_meters and x_meters. Default is the LA County.
        steps: number of bins to create along the y_meters and x_meters.
        study_area_mbr and steps are used to divide the study area. The study area will be divided into steps X steps grids for R tree queries and efficient PSCO_2 mining.
  
        '''
        self.K = K
        self.remove_id = remove_id
        self.diff_privacy = diff_privacy
        self.epsilon_location = epsilon_location
        self.epsilon_time = epsilon_time
        self.sensitivity_meters = sensitivity_meters
        self.sensitivity_seconds = sensitivity_seconds
        self.apply_time_noise = apply_time_noise
        
        self.threshold_mbr_distance = threshold_mbr_distance
        
        self.max_area = max_area
        self.min_area = min_area
        self.x_replace_range = x_replace_range
        self.y_replace_range = y_replace_range
        
        self.threshold_count = threshold_count
        self.time_order = time_order
        
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.largest_cluster_min_points = largest_cluster_min_points

        self.x_bins = x_bins
        self.y_bins = y_bins
        

    def preprocess_traj(self, raw_traj):

        agents = relative_coordinates(raw_traj)
        if self.diff_privacy == True: 
            agents = noise_trajectory_batch_meters(agents, self.epsilon_location, self.epsilon_time, self.sensitivity_meters, self.sensitivity_seconds, self.apply_time_noise)
        agents['utc_time'] = pd.to_datetime(agents['utc_timestamp'], unit='s', utc=True)
        agents['la_time'] = agents['utc_time'].dt.tz_convert('America/Los_Angeles')
        agents = agents[['caid', 'x_meters', 'y_meters', 'la_time']].copy()
        agents['hour'] = agents['la_time'].dt.hour
        agents['day'] = agents['la_time'].dt.dayofweek
        agents.drop(columns = ['la_time'], inplace=True)
        agents['IsWeekend'] = agents['day'].apply(lambda x: 1 if x>=5 else 0)
        agents.drop(columns = ['day'], inplace=True)
        agents = agents.dropna()
        agents = agents.reset_index(drop=True)
        return agents   
    
    
    def preprocess_traj_synthetic(self, agents, time_step):

        if self.diff_privacy == True: 
            agents = noise_trajectory_batch_meters(agents, self.epsilon_location, self.epsilon_time, self.sensitivity_meters, self.sensitivity_seconds, self.apply_time_noise)
        agents = add_isweekend_hour(agents, time_step)
        return agents  

        
    def mine_PSCO_1(self, agents):
        PSCO_1 = {}
        for caid, data in agents.groupby('caid'):
            day_hour_grouped = {f"weekend_{weekend}_hour_{hour}": sub_group.reset_index(drop=True) for (weekend, hour), sub_group in data.groupby(['IsWeekend', 'hour']) if len(sub_group) >= self.largest_cluster_min_points} 
            if len(day_hour_grouped) >= self.threshold_count: # remove agents without enough time-windows before dbscan
                traj = []
                for time_slot in self.time_order:
                    if time_slot in day_hour_grouped:
                        locations = day_hour_grouped[time_slot]
                        largest_cluster = most_visited_cluster(locations, self.dbscan_eps, self.dbscan_min_samples, self.largest_cluster_min_points) # largest_clusters with less than 5 points are returned empty dataframes
                        if not largest_cluster.empty: # only largest_clusters with >= 5 points are kept
                            mbr = calculate_mbr(largest_cluster)
                            mbr_area = calculate_mbr_area(mbr)
                            if mbr_area <= self.max_area:
                                traj.append((time_slot, mbr))
                if len(traj) >= self.threshold_count:
                        PSCO_1[caid] = traj
        return PSCO_1        

    def get_degree_offsets(self, threshold_mbr_distance):
        '''
        Find the buffer level based on the threshold distance
        '''

        y_offset = threshold_mbr_distance  * 2  
        x_offset = threshold_mbr_distance  * 2  
        return y_offset, x_offset    



    def create_mbr_grid(self, x_min, y_min, x_max, y_max, x_step, y_step):
        '''
        threshold_mbr_distance : the maximum distance between nearby mbrs, which is used to create a buffer so that there are no missing co-locations. 
        '''
        y_offset, x_offset = self.get_degree_offsets(self.threshold_mbr_distance)
        mbrs = []  
    
        y = y_min
        while y < y_max:
            x = x_min
            while x < x_max:
    
                mbr = (x - x_offset, y - y_offset, x + x_step + x_offset, y + y_step + y_offset)
                mbrs.append(mbr)
                x += x_step
            y += y_step
        return mbrs



    def mine_PSCO_2(self, PSCO_1):   

        PSCO_2 = {}
        transformed_PSCO_1 = transform_PSCO(PSCO_1, self.time_order)
        for time_window, agent_mbrs in transformed_PSCO_1.items():
            tree = index.Index()
            for idx, (agent_id, mbr) in enumerate(agent_mbrs.items()):
 
                mbr_tuple = mbr_dict_to_tuple(mbr) 
                tree.insert(idx, mbr_tuple, obj=agent_id)
            x_min, y_min, x_max, y_max = tree.bounds 
            x_step = (x_max - x_min) / self.x_bins
            y_step = (y_max - y_min) / self.y_bins
            grid_cell_mbrs = self.create_mbr_grid(x_min, y_min, x_max, y_max, x_step, y_step)
            g2_cotravel_time = []
            for grid in grid_cell_mbrs:
               
                results = list(tree.intersection(grid, objects=True))
                agent_ids = [result.object for result in results]
                if not agent_ids:  
                    continue      
                grid_cotravellers = two_agent_colocations(time_window, transformed_PSCO_1, agent_ids, self.threshold_mbr_distance, self.max_area)
                g2_cotravel_time.append(grid_cotravellers)
            g2_cotravel_time_df = pd.concat(g2_cotravel_time, axis=0, ignore_index=True)
            g2_cotravel_time_df_no_dup = g2_cotravel_time_df.drop_duplicates(subset='group', keep = 'first')
            if g2_cotravel_time_df_no_dup.empty:
                return {}
            
            g2_cotravel_time_dict = g2_cotravel_time_df_no_dup.set_index('group')['grouped_points_mbr'].to_dict()
            for group, location_mbr in g2_cotravel_time_dict.items():
                if group not in PSCO_2:
                    PSCO_2[group] = []
                PSCO_2[group].append((time_window, location_mbr))   
    
        PSCO_2 = {k: v for k, v in PSCO_2.items() if len(v) >= self.threshold_count}
    
        if PSCO_2:
            return PSCO_2
        else:
            return {}
    



    def generate_candidates(self, PSCO_k_groups):

        if len(PSCO_k_groups) < 2:
            return {}


        PSCO_k_groups_list = sorted(PSCO_k_groups)
    
        prefix_groups = {}
        for group in PSCO_k_groups_list:
            prefix = group[:-1]
            prefix_groups.setdefault(prefix, []).append(group)

        candidates = {}
        
        for prefix, groups in prefix_groups.items():
            n = len(groups)
            if n > 1:
                for i in range(n):
                    for j in range(i + 1, n):
                        group1, group2 = groups[i], groups[j]
                        candidate = group1 + (group2[-1],)
                        candidates[candidate] = (group1, group2)

        return candidates


    def mine_k_1(self, PSCO_k_groups):
        """
        Pipeline: generate candidates, prune them, and evaluate each 
        PSCO_k_groups: a dictionary of colocation groups of all time_windows
        Mining only K+1
        """
        if PSCO_k_groups == {}:
            return "No more groups"
        transformed_PSCO_k = transform_PSCO(PSCO_k_groups, self.time_order)
        num_windows = len(transformed_PSCO_k)
        PSCO_k = {} 
        
        i = 0
        for time_window, groups in transformed_PSCO_k.items():
            i += 1
            if i <= (num_windows - self.threshold_count + 1):
                generated_can =  self.generate_candidates(groups)
                init_pruned = prune_candidates(generated_can, groups)
                area_pruned = check_colocation_area(init_pruned, groups, self.max_area)
                for group, location_mbr in area_pruned.items():
                    if group not in PSCO_k:
                        PSCO_k[group] = []
                    PSCO_k[group].append((time_window, location_mbr))
                
            else:
                generated_can =  self.generate_candidates(groups)
                existing_can = {k: v for k, v in generated_can.items() if k in PSCO_k} # keep only the existing groups
                init_pruned = prune_candidates(existing_can, groups)
                area_pruned = check_colocation_area(init_pruned, groups, self.max_area)
                for group, location_mbr in area_pruned.items():
                    PSCO_k[group].append((time_window, location_mbr))
                PSCO_k = {k: v for k, v in PSCO_k.items() if len(v) >= i - (num_windows - self.threshold_count)}

        if PSCO_k:
            return PSCO_k
        else:
            return "No more groups"
    
    def mine_PSCO_k(self, PSCO_k_groups):
        k=2
        all_groups = {}
        all_groups['PSCO_2'] = PSCO_k_groups
        PSCO_k_groups_loop = PSCO_k_groups
        while k > 0:
            PSCO_k_1 = self.mine_k_1(PSCO_k_groups_loop)
            if PSCO_k_1 == "No more groups":
                
                # K-anonymity
                K_filtered_dict = {key: value for key, value in all_groups.items() if int(key.split('_')[1]) >= self.K}
                if self.remove_id == "Yes":
                    K_filtered_dict = {outer_key: {f"{outer_key}_{i + 1}": value for i, value in enumerate(inner_dict.values())} for outer_key, inner_dict in K_filtered_dict.items()}
                else:
                    return K_filtered_dict
                return K_filtered_dict
            else:
                key = 'PSCO_' + str(k+1)
                all_groups[key] = PSCO_k_1
                k += 1
                PSCO_k_groups_loop = PSCO_k_1
                
                
                
    def mine_PSCO_k_inter(self, PSCO_k_groups):
        '''
        Intermediate function for refining

        '''
        k=2
        all_groups = {}
        all_groups['PSCO_2'] = PSCO_k_groups
        PSCO_k_groups_loop = PSCO_k_groups
        while k > 0:
            PSCO_k_1 = self.mine_k_1(PSCO_k_groups_loop)
            if PSCO_k_1 == "No more groups": 
                
                # K-anonymity
                K_filtered_dict = {key: value for key, value in all_groups.items() if int(key.split('_')[1]) >= self.K}
                return K_filtered_dict
            else:
                key = 'PSCO_' + str(k+1)
                all_groups[key] = PSCO_k_1
                k += 1
                PSCO_k_groups_loop = PSCO_k_1
                


    
    
    def find_contained_groups(self, PSCO_small, PSCO_large):


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

    
    
    
    
    def refine_PSCO_k(self, PSCO):
        '''
        Remove smaller PSCO that completely contained in a larger PSCO
        PSCO : A dictionary of all sizes of PSCOs
        remove smaller groups that contained in larger groups.
        '''
        filtered_PSCOs = {}
        largest_group_size = len(PSCO) + self.K - 1
        
        # K-anonymity
        for i in range(self.K, largest_group_size):
            PSCO_small = PSCO[f"PSCO_{i}"]
            PSCO_large = PSCO[f"PSCO_{i+1}"]
            groups_contained = self.find_contained_groups(PSCO_small, PSCO_large)
            filtered_PSCOs[f"PSCO_{i}"]={key: value for key, value in PSCO_small.items() if key not in groups_contained}
        filtered_PSCOs[f"PSCO_{largest_group_size}"] = PSCO[f"PSCO_{largest_group_size}"] 
        # cloak
        filtered_PSCOs = cloak(filtered_PSCOs, self.x_replace_range, self.y_replace_range, self.min_area)

        if self.remove_id == "Yes": 
            filtered_PSCOs= {outer_key: {f"{outer_key}_{i + 1}": value for i, value in enumerate(inner_dict.values())} for outer_key, inner_dict in filtered_PSCOs.items()}
        else:
            return filtered_PSCOs
        # remove empty dictionaries
        filtered_PSCOs = {k: v for k, v in filtered_PSCOs.items() if v != {}}
        return filtered_PSCOs
    
    def mine_refine_PSCO_k(self, PSCO_k_groups):
        '''
        Remove smaller co-traveling groups that contained completely in a larger co-traveling group
        '''
        PSCO_all = self.mine_PSCO_k_inter(PSCO_k_groups)
        PSCO_refined = self.refine_PSCO_k(PSCO_all)
        PSCO_refined = {k: v for k, v in PSCO_refined.items() if v != {}}
        return PSCO_refined
    




############################## Original ############################################################
####################################################################################################   


class RCPD_miner_original:
    def __init__(self, K = 2, remove_id = "Yes", diff_privacy = True, epsilon_location=0.5, epsilon_time=0.5, sensitivity_meters=5, sensitivity_seconds=100, apply_time_noise=True, threshold_mbr_distance = 20, max_area = 12100, min_area = 100, x_replace_range = (10, 20), y_replace_range = (10, 20), threshold_count = 3, time_order = [f"weekend_{i}_hour_{j}" for i in range(2) for j in range(24)], dbscan_eps = 20, dbscan_min_samples = 3, largest_cluster_min_points = 5):
        '''
        The PSCO naive miner input k-size co-traveling groups and output (k+1) size co-traveling groups
        dfs: a dictionary containing location data for each agent
        min_area (m^2): the threshold of mbr area that decides whether the mbr is too small that needs to be cloaked.
        x_replace_range (m): replace the old mbr with a new mbr with an x-length randomly picked in this range and the original mbr center. 
        y_replace_range (m): replace the old mbr with a new mbr with a y-length randomly picked in this range and the original mbr center.
        '''
        self.K = K
        self.remove_id = remove_id
        self.diff_privacy = diff_privacy
        self.epsilon_location = epsilon_location
        self.epsilon_time = epsilon_time
        self.sensitivity_meters = sensitivity_meters
        self.sensitivity_seconds = sensitivity_seconds
        self.apply_time_noise = apply_time_noise
        
        self.threshold_mbr_distance = threshold_mbr_distance
        
        self.max_area = max_area
        self.min_area = min_area
        self.x_replace_range = x_replace_range
        self.y_replace_range = y_replace_range
        
        self.threshold_count = threshold_count
        self.time_order = time_order
        
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.largest_cluster_min_points = largest_cluster_min_points
        
    
    
    
    def preprocess_traj(self, raw_traj):

        agents = relative_coordinates(raw_traj)
        if self.diff_privacy == True: 
            agents = noise_trajectory_batch_meters(agents, self.epsilon_location, self.epsilon_time, self.sensitivity_meters, self.sensitivity_seconds, self.apply_time_noise)
        agents['utc_time'] = pd.to_datetime(agents['utc_timestamp'], unit='s', utc=True)
        agents['la_time'] = agents['utc_time'].dt.tz_convert('America/Los_Angeles')
        agents = agents[['caid', 'x_meters', 'y_meters', 'la_time']].copy()
        agents['hour'] = agents['la_time'].dt.hour
        agents['day'] = agents['la_time'].dt.dayofweek
        agents.drop(columns = ['la_time'], inplace=True)
        agents['IsWeekend'] = agents['day'].apply(lambda x: 1 if x>=5 else 0)
        agents.drop(columns = ['day'], inplace=True)
        agents = agents.dropna()
        agents = agents.reset_index(drop=True)
        return agents   
    
    
    def preprocess_traj_synthetic(self, agents, time_step):
        # differential privacy
        if self.diff_privacy == True: 
            agents = noise_trajectory_batch_meters(agents, self.epsilon_location, self.epsilon_time, self.sensitivity_meters, self.sensitivity_seconds, self.apply_time_noise)
        agents = add_isweekend_hour(agents, time_step)
        return agents  
    
    
    
        
    def mine_PSCO_1(self,agents):
        PSCO_1 = {}
        for caid, data in agents.groupby('caid'):
            if 'IsWeekend' not in data.columns:
                print(f" {caid} Missing 'IsWeekend' column!")
                print("Available columns:", data.columns)
                print("First few rows:\n", data.head())
                continue
            day_hour_grouped = {f"weekend_{weekend}_hour_{hour}": sub_group.reset_index(drop=True) for (weekend, hour), sub_group in data.groupby(['IsWeekend', 'hour'])}
            traj = []
            for time_slot in self.time_order:
                if time_slot in day_hour_grouped:
                    locations = day_hour_grouped[time_slot]
                    largest_cluster = most_visited_cluster(locations, self.dbscan_eps, self.dbscan_min_samples, self.largest_cluster_min_points) # largest_clusters with less than 5 points are returned empty dataframes
                    if not largest_cluster.empty: # only largest_clusters with >= 5 points are kept
                        mbr = calculate_mbr(largest_cluster)
                        mbr_area = calculate_mbr_area(mbr)
                        if mbr_area <= self.max_area:
                            traj.append((time_slot, mbr))
            if len(traj) >= self.threshold_count:
                    PSCO_1[caid] = traj
        return PSCO_1


    def mine_PSCO_2(self, PSCO_1):
    
        PSCO_2 = {}
        transformed_PSCO_1 = transform_PSCO(PSCO_1, self.time_order)
        for time_window, agent_mbrs in transformed_PSCO_1.items():
            agent_ids = list(agent_mbrs.keys())
            colocations = two_agent_colocations(time_window, transformed_PSCO_1, agent_ids, self.threshold_mbr_distance, self.max_area)
            colocations_no_dup = colocations.drop_duplicates(subset='group', keep = 'first')
            if colocations_no_dup.empty: 
                return {}   
            colocations_dict = colocations_no_dup.set_index('group')['grouped_points_mbr'].to_dict()
            for group, location_mbr in colocations_dict.items():
                if group not in PSCO_2:
                    PSCO_2[group] = []
                PSCO_2[group].append((time_window, location_mbr))   
    
        PSCO_2 = {k: v for k, v in PSCO_2.items() if len(v) >= self.threshold_count}
    
        if PSCO_2:
            return PSCO_2
        else:
            return {}


    def generate_candidates(self, PSCO_k_groups):

        PSCO_k_groups_list = list(PSCO_k_groups)
        if len(PSCO_k_groups_list) < 2:
            return {}
        PSCO_k_groups_list = sorted(PSCO_k_groups_list)
        candidates = {}
        for i in range(len(PSCO_k_groups_list)):
            group1 = PSCO_k_groups_list[i]
            for j in range(i + 1, len(PSCO_k_groups_list)):
                group2 = PSCO_k_groups_list[j]
                if group1[:-1] == group2[:-1]:  
                    candidate = group1 + (group2[-1],)  
                    candidates[candidate] = (group1, group2)  
        return candidates





    def mine_k_1(self, PSCO_k_groups):
        """
        Pipeline: generate candidates, prune them, and evaluate each 
        PSCO_k_groups: a dictionary of colocation groups of all time_windows
        Mining only K+1
        """
        
        if PSCO_k_groups == {}:
            return "No more groups"
        transformed_PSCO_k = transform_PSCO(PSCO_k_groups, self.time_order)
        
        PSCO_k = {} # a dictionary with time windows as keys 
        for time_window, groups in transformed_PSCO_k.items():
            generated_can =  self.generate_candidates(groups)
            init_pruned = prune_candidates(generated_can, groups)
            area_pruned = check_colocation_area(init_pruned, groups, self.max_area)
            for group, location_mbr in area_pruned.items():
                if group not in PSCO_k:
                    PSCO_k[group] = []
                PSCO_k[group].append((time_window, location_mbr))   

        PSCO_k = {k: v for k, v in PSCO_k.items() if len(v) >= self.threshold_count}

        if PSCO_k:
            return PSCO_k
        else:
            return "No more groups"

    
    def mine_PSCO_k(self, PSCO_k_groups):
        k=2
        all_groups = {}
        all_groups['PSCO_2'] = PSCO_k_groups
        PSCO_k_groups_loop = PSCO_k_groups
        while k > 0:
            PSCO_k_1 = self.mine_k_1(PSCO_k_groups_loop)
            if PSCO_k_1 == "No more groups":
                
                # K-anonymity
                K_filtered_dict = {key: value for key, value in all_groups.items() if int(key.split('_')[1]) >= self.K}
                if self.remove_id == "Yes":
                    K_filtered_dict = {outer_key: {f"{outer_key}_{i + 1}": value for i, value in enumerate(inner_dict.values())} for outer_key, inner_dict in K_filtered_dict.items()}
                else:
                    return K_filtered_dict
                return K_filtered_dict
            else:
                key = 'PSCO_' + str(k+1)
                all_groups[key] = PSCO_k_1
                k += 1
                PSCO_k_groups_loop = PSCO_k_1
                                
                
                
    def mine_PSCO_k_inter(self, PSCO_k_groups):
        '''
        Intermediate function for refining

        '''
        k=2
        all_groups = {}
        all_groups['PSCO_2'] = PSCO_k_groups
        PSCO_k_groups_loop = PSCO_k_groups
        while k > 0:
            PSCO_k_1 = self.mine_k_1(PSCO_k_groups_loop)
            if PSCO_k_1 == "No more groups": 
                return all_groups
            else:
                key = 'PSCO_' + str(k+1)
                all_groups[key] = PSCO_k_1
                k += 1
                PSCO_k_groups_loop = PSCO_k_1

   
    

    def find_contained_groups(self, PSCO_small, PSCO_large):

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

    
    
    def refine_PSCO_k(self, PSCO):
        '''
        Remove smaller PSCO that completely contained in a larger PSCO
        PSCO : A dictionary of all sizes of PSCOs
        remove smaller groups that contained in larger groups.
        '''
        filtered_PSCOs = {}
        largest_group_size = len(PSCO) + 1
        for i in range(2,largest_group_size):
            PSCO_small = PSCO[f"PSCO_{i}"]
            PSCO_large = PSCO[f"PSCO_{i+1}"]
            groups_contained = self.find_contained_groups(PSCO_small, PSCO_large)
            filtered_PSCOs[f"PSCO_{i}"]={key: value for key, value in PSCO_small.items() if key not in groups_contained}
        filtered_PSCOs[f"PSCO_{largest_group_size}"] = PSCO[f"PSCO_{largest_group_size}"] 
        
        # K-anonymity
        filtered_PSCOs = {key: value for key, value in filtered_PSCOs.items() if int(key.split('_')[1]) >= self.K}
        # cloak
        filtered_PSCOs = cloak(filtered_PSCOs, self.x_replace_range, self.y_replace_range, self.min_area)
        if self.remove_id == "Yes": 
            filtered_PSCOs= {outer_key: {f"{outer_key}_{i + 1}": value for i, value in enumerate(inner_dict.values())} for outer_key, inner_dict in filtered_PSCOs.items()}
        else:
            return filtered_PSCOs
        filtered_PSCOs = {k: v for k, v in filtered_PSCOs.items() if v != {}}
        return filtered_PSCOs
    
    def mine_refine_PSCO_k(self, PSCO_k_groups):
        '''
        Remove smaller co-traveling groups that contained completely in a larger co-traveling group
        '''
        PSCO_all = self.mine_PSCO_k_inter(PSCO_k_groups)
        PSCO_refined = self.refine_PSCO_k(PSCO_all)
        PSCO_refined = {k: v for k, v in PSCO_refined.items() if v != {}}
        return PSCO_refined
    

