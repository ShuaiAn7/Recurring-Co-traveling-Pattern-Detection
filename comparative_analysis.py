# -*- coding: utf-8 -*-
"""
@author: Shuai An
"""


import pandas as pd
import time
import joblib
from RCPD_miner import RCPD_miner_faster, RCPD_miner_original
import random
random.seed(2025)
from itertools import combinations
from haversine import haversine, Unit



agents = pd.read_parquet('data/input/agents_500.parquet')


####################  RCPD  ########################################


######## faster ###############################################################

start_time = time.time()
pm_faster = RCPD_miner_faster(K=4)
psco_faster_1 = pm_faster.mine_PSCO_1(agents)
end_time = time.time()
running_time = (end_time-start_time)/60
print(f"PSCO_1 faster mining time mins: {running_time}")

start_time = time.time()
psco_faster_2 = pm_faster.mine_PSCO_2(psco_faster_1)
end_time = time.time()
running_time = (end_time-start_time)/60
print(f"PSCO_2 faster mining time mins: {running_time}")
print("Faster: ", psco_faster_2)


#########Original##############################################################

start_time = time.time()
pm_original = RCPD_miner_original(K=4)
psco_original_1 = pm_original.mine_PSCO_1(agents)
end_time = time.time()
running_time = (end_time-start_time)/60
print(f"PSCO_1 original mining time mins: {running_time}")

start_time = time.time()
psco_original_2 = pm_original.mine_PSCO_2(psco_original_1)
end_time = time.time()
running_time = (end_time-start_time)/60
print(f"PSCO_2 original mining time mins: {running_time}")
print("Original: ", psco_original_2)




############Test Results######################################################

if psco_original_1 == psco_faster_1:
    print("The two methods result in the same PSCO_1 !")
else:
    print("Need to debug...")
    
    
if psco_original_2 == psco_faster_2:
    print("The two methods result in the same PSCO_2 !")
else:
    print("Need to debug...")
    
    
###############################  MDCOP  ######################################



def apriori_gen(Pk):

    Ck_plus_1 = []
    Pk_set = set(Pk)
    for p in Pk:
        for q in Pk:
            if p[:-1] == q[:-1] and p[-1] < q[-1]:
                candidate = p + (q[-1],)
                Ck_plus_1.append(candidate)
    pruned_candidates = []
    for c in Ck_plus_1:
        subsets = combinations(c, len(c) - 1)
        
        if all(tuple(sorted(subset)) in Pk_set for subset in subsets):
            pruned_candidates.append(c)
    return pruned_candidates


def mine_colocation_1(location_data, min_num_instances):

    filtered_location_data = location_data[location_data.groupby('caid')['caid'].transform('size') >= min_num_instances].copy()
    filtered_location_data['row_number'] = filtered_location_data.groupby('caid').cumcount() + 1
    filtered_location_data['instance_id'] = filtered_location_data['caid'].astype(str) + "_" + filtered_location_data['row_number'].astype(str)
    filtered_location_data = filtered_location_data.drop(['row_number'], axis=1)
    filtered_location_data = filtered_location_data.sort_values(by='instance_id').reset_index(drop=True)
    caid_instance_count = filtered_location_data.groupby('caid').size().to_dict()
    return filtered_location_data, caid_instance_count

    

def participation_index(init_instances, caid_instance_count):

    participation_indices = {}

    for caid_pair, instances in init_instances.items():
        participation_ratios = []

        for caid in caid_pair:

            total_instances = caid_instance_count[caid]
            
            participating_instances = instances[caid].unique()
            
            participation_ratio = len(participating_instances) / total_instances
            participation_ratios.append(participation_ratio)
        
        participation_indices[caid_pair] = min(participation_ratios)
    return participation_indices

def mine_colocation_2(location_data, distance_threshold, caid_instance_count, PI_threshold):   
    instances_init_2 = {}

    for (idx1, row1), (idx2, row2) in combinations(location_data.iterrows(), 2):

        if row1["caid"] == row2["caid"]:
            continue

        coord1 = (row1["latitude"], row1["longitude"])
        coord2 = (row2["latitude"], row2["longitude"])
        distance = haversine(coord1, coord2, unit=Unit.METERS)
        
        if distance < distance_threshold:

            instance_pair = tuple(sorted([row1["instance_id"], row2["instance_id"]]))
    
            caid1 = row1["caid"]
            caid2 = row2["caid"]
    
            group_key = tuple(sorted([str(caid1), str(caid2)]))
    
            if group_key not in instances_init_2:
                instances_init_2[group_key] = []
            instances_init_2[group_key].append(instance_pair)
    
    instances_2 = {
        colocation: pd.DataFrame(instances, columns=colocation)
        for colocation, instances in instances_init_2.items()}

    participation_indices_2 = participation_index(instances_2, caid_instance_count)
    
    R = {colocation: set(instances) for colocation, instances in instances_init_2.items() 
         if participation_indices_2.get(colocation, 0) >= PI_threshold}
    
    table_instances_2 = {
        colocation: instances
        for colocation, instances in instances_2.items()
        if participation_indices_2.get(colocation, 0) >= PI_threshold}
    
    if not table_instances_2:
        return "No R", "No Instances", "No Colocations"
    else:
        colocation_2 = list(table_instances_2.keys())
        return R, table_instances_2, colocation_2
    


def generate_k_1_instance(C_k_1, table_instances_k, R):

    instances_k_1 = {}

    for c in C_k_1:

        colocation_k_1 = c[:-1]  
        colocation_k_2 = c[:-2] + (c[-1],)  
        new_feature_pair = c[-2:]

        if not new_feature_pair in R:
            continue


        table_1 = table_instances_k.get(colocation_k_1, pd.DataFrame())  
        table_2 = table_instances_k.get(colocation_k_2, pd.DataFrame())  

        if table_1.empty or table_2.empty:
            continue

        sort_columns = list(table_1.columns)[:-1]
        table_1 = table_1.sort_values(by=sort_columns)
        table_2 = table_2.sort_values(by=sort_columns)

        merged = pd.merge(
            table_1,
            table_2,
            on=sort_columns)

        k_1_instances = merged[
            merged.apply(
                lambda row: tuple(row.iloc[-2:]) in R[new_feature_pair],
                axis=1,
            )
        ]
        if not k_1_instances.empty:
            instances_k_1[c] = k_1_instances

    return instances_k_1



def mine_colocation_k_1(table_instances_k, R, caid_instance_count, PI_threshold):

    k_groups = list(table_instances_k.keys())
    k_1_can = apriori_gen(k_groups)

    k_1_instances = generate_k_1_instance(k_1_can, table_instances_k, R)
    k_1_PI = participation_index(k_1_instances, caid_instance_count)
    table_instances_k_1 = {colocation: instances for colocation, instances in k_1_instances.items() if k_1_PI.get(colocation, 0) >= PI_threshold}
    if not table_instances_k_1:
        return "No Instances", "No Colocations"
    else:
        k_1_colocation = list(table_instances_k_1.keys())
        return table_instances_k_1, k_1_colocation
    
    

def mine_colocations(location_data, min_num_instances, distance_threshold, PI_threshold):
    colocations = {}
    instance_1, caid_instance_count = mine_colocation_1(location_data, min_num_instances)
    R, instances_2, colocation_2 = mine_colocation_2(instance_1, distance_threshold, caid_instance_count, PI_threshold) 
    if colocation_2 == "No Colocations":
        return "No Colocations"
    elif len(colocation_2) == 1:
        colocations['colocation_2'] = colocation_2
        return colocations
    colocations['colocation_2'] = colocation_2
    colocation_k_loop = instances_2
    k = 2
    while k > 0:
        instances_k, colocation_k = mine_colocation_k_1(colocation_k_loop, R, caid_instance_count, PI_threshold)
        if colocation_k == "No Colocations":
            return colocations
        else:
            key = "colocation_" + str(k+1)
            colocations[key] = colocation_k
            k += 1
            colocation_k_loop = instances_k


def mine_colocations_pruning(location_data, min_num_instances, distance_threshold, PI_threshold, colocation_count_dict):
    colocations_filtered = {}
    instance_1, caid_instance_count = mine_colocation_1(location_data, min_num_instances)
    R, instances_2, colocation_2 = mine_colocation_2(instance_1, distance_threshold, caid_instance_count, PI_threshold) 
    if colocation_2 == "No Colocations":
        return "No Colocations"
    colocation_2_filtered = [item for item in colocation_2 if item in colocation_count_dict['colocation_2']]
    if len(colocation_2_filtered) == 0:
        return "No Colocations"
    elif len(colocation_2_filtered) == 1:
        colocations_filtered['colocation_2'] = colocation_2_filtered
        return colocations_filtered
    colocations_filtered['colocation_2'] = colocation_2_filtered
    colocation_k_loop = {group: data for group, data in instances_2.items() if group in colocation_count_dict['colocation_2']}

    k = 2
    while k > 0:
        instances_k, colocation_k = mine_colocation_k_1(colocation_k_loop, R, caid_instance_count, PI_threshold)
        if colocation_k == "No Colocations":
            return colocations_filtered
        else:
            key = "colocation_" + str(k+1)
            colocation_k_filtered = [item for item in colocation_k if item in colocation_count_dict[key]]
            colocations_filtered[key] = colocation_k_filtered
            colocation_k_loop = {group: data for group, data in instances_k.items() if group in colocation_count_dict[key]}
            k += 1
            
        


time_order = [f"weekend_{i}_hour_{j}" for i in range(2) for j in range(24)]
def MDCOPfast(agents, min_num_instances=5, distance_threshold=10, PI_threshold=0.5, threshold_count=3):
    day_hour_grouped = {f"weekend_{weekend}_hour_{hour}": sub_group.reset_index(drop=True) for (weekend, hour), sub_group in agents.groupby(['IsWeekend', 'hour'])}
    colocations_all = {}
    for i in range(len(time_order)):
        if i <= (len(time_order) - threshold_count + 1):
            time_window = time_order[i]
            if time_window not in day_hour_grouped:
                continue
            location_data = day_hour_grouped[time_window]
            colocations = mine_colocations(location_data, min_num_instances, distance_threshold, PI_threshold)
            if colocations == "No Colocations":
                continue
            for k, groups in colocations.items():
                if k not in colocations_all:
                    colocations_all[k] = {}
                for group in groups:
                    if group not in colocations_all[k]:
                        colocations_all[k][group] = []
                    colocations_all[k][group].append(time_window)
        else:
            j =  i - (len(time_order) - threshold_count + 1)
            colocations_all = {
        group: {key: value for key, value in sub_dict.items() if len(value) >= j} for group, sub_dict in colocations_all.items()}
            time_window = time_order[i]
            if time_window not in day_hour_grouped:
                continue
            location_data = day_hour_grouped[time_window]
            colocations_filtered = mine_colocations_pruning(location_data, min_num_instances, distance_threshold, PI_threshold, colocations_all)
            if colocations_filtered == "No Colocations":
                continue
            for k, groups in colocations_filtered.items():
                for group in groups:
                    colocations_all[k][group].append(time_window)
    colocations_all = {
group: {key: value for key, value in sub_dict.items() if len(value) >= threshold_count} for group, sub_dict in colocations_all.items()}
    return colocations_all



def MDCOP(agents, min_num_instances=5, distance_threshold=10, PI_threshold=0.5, threshold_count=3):
    day_hour_grouped = {f"weekend_{weekend}_hour_{hour}": sub_group.reset_index(drop=True) for (weekend, hour), sub_group in agents.groupby(['IsWeekend', 'hour'])}
    colocations_all = {}
    for i in range(len(time_order)):
        if i <= (len(time_order) - threshold_count + 1):
            time_window = time_order[i]
            if time_window not in day_hour_grouped:
                continue
            location_data = day_hour_grouped[time_window]
            colocations = mine_colocations(location_data, min_num_instances, distance_threshold, PI_threshold)
            if colocations == "No Colocations":
                continue
            for k, groups in colocations.items():
                if k not in colocations_all:
                    colocations_all[k] = {}
                for group in groups:
                    if group not in colocations_all[k]:
                        colocations_all[k][group] = []
                    colocations_all[k][group].append(time_window)
    colocations_all = {
    group: {key: value for key, value in sub_dict.items() if len(value) >= threshold_count} for group, sub_dict in colocations_all.items()}
    return colocations_all


agents = pd.read_parquet('data/input/part_1.parquet')

ids = agents['caid'].unique()

ids = list(ids)

sample_ids = random.sample(ids, 500)

agents_trial = agents[agents['caid'].isin(sample_ids)].copy()


min_num_instances = 5
distance_threshold = 20 # m 
PI_threshold = 0.5
threshold_count = 3


start_time = time.time()
mdcop = MDCOP(agents_trial, min_num_instances, distance_threshold, PI_threshold, threshold_count)
end_time = time.time()
print("MDCOP running time mins: ", (end_time - start_time)/60)


joblib.dump(mdcop, "data/output/mdcop_500.joblib")

