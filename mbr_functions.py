# -*- coding: utf-8 -*-
"""
@author: Shuai An
"""


import numpy as np
import random
random.seed(2025)
import copy

def mbr_dict_to_tuple(mbr):
    ys = [mbr[corner]["y_meters"] for corner in mbr]
    xs = [mbr[corner]["x_meters"] for corner in mbr]
    return (min(xs), min(ys), max(xs), max(ys))  # (left, bottom, right, top)


def calculate_mbr(y_x_points):
    """
    Calculate the Minimum Bounding Rectangle (MBR) for a group of y_meters and x_meters points.
    
    Parameters:
    y_x_points (pd.DataFrame): DataFrame with columns 'y_meters' and 'x_meters' containing the points.
    
    Returns:
    dict: A dictionary with the coordinates of the corners of the MBR.
    """
    if not y_x_points.empty:
        min_y = y_x_points['y_meters'].min()
        max_y = y_x_points['y_meters'].max()
        min_x = y_x_points['x_meters'].min()
        max_x = y_x_points['x_meters'].max()
        mbr = {
            'bottom_left': {'y_meters': min_y,'x_meters': min_x},
            'bottom_right': {'y_meters': min_y, 'x_meters': max_x},
            'top_left': {'y_meters': max_y, 'x_meters': min_x},
            'top_right': {'y_meters': max_y, 'x_meters': max_x}
        }
        return mbr
    else:
        return {}
    
def calculate_mbr_center(mbr):
    center_x = (mbr['bottom_left']['x_meters'] + mbr['top_right']['x_meters']) / 2
    center_y = (mbr['bottom_left']['y_meters'] + mbr['top_right']['y_meters']) / 2
    return {'y_center': center_y, 'x_center': center_x}
  

def create_mbr_from_center(y_center, x_center, y_length, x_length):
    half_y = y_length / 2
    half_x = x_length / 2
    
    return {
        'bottom_left': {'y_meters': y_center - half_y, 'x_meters': x_center - half_x},
        'bottom_right': {'y_meters': y_center - half_y, 'x_meters': x_center + half_x},
        'top_left': {'y_meters': y_center + half_y, 'x_meters': x_center - half_x},
        'top_right': {'y_meters': y_center + half_y, 'x_meters': x_center + half_x}}

    
def calculate_mbr_area(mbr):
    if not mbr:
        return float('inf')
    
    else:
        y_min = mbr['bottom_left']['y_meters']
        y_max = mbr['top_left']['y_meters']
        x_min = mbr['bottom_left']['x_meters']
        x_max = mbr['bottom_right']['x_meters']
        y_diff = y_max - y_min
        x_diff = x_max - x_min
        area = y_diff * x_diff
        return area


def mbr_area_stats(nested_dict):
    areas = []
    zero_area_count = 0
    
    for main_key, subdict in nested_dict.items():
        for sub_key, mbr_list in subdict.items():
            for time_slot, mbr in mbr_list:
                area = calculate_mbr_area(mbr)
                areas.append(area)
                if area == 0:
                    zero_area_count += 1
    
    areas = np.array(areas)     
    stats = {
        'total count': len(areas),
        'zero_area_count': zero_area_count,
        'min_area': np.min(areas) if areas.size > 0 else None,
        'max_area': np.max(areas) if areas.size > 0 else None,
        'mean_area': np.mean(areas) if areas.size > 0 else None,
        'median_area': np.median(areas) if areas.size > 0 else None} 
    return stats


def cloak(nested_dict, x_length_range, y_length_range, area_threshold):
    new_nested_dict = copy.deepcopy(nested_dict)
    
    replaced_count = 0
    total_count = 0
    
    for main_key, subdict in new_nested_dict.items():
        for sub_key, mbr_list in subdict.items():
            for idx, (time_slot, mbr) in enumerate(mbr_list):
                total_count += 1
                area = calculate_mbr_area(mbr)
                if area < area_threshold:
                    
                    y_center = (mbr['bottom_left']['y_meters'] + mbr['top_right']['y_meters']) / 2
                    x_center = (mbr['bottom_left']['x_meters'] + mbr['top_right']['x_meters']) / 2
                    
                  
                    random_x_length = random.uniform(x_length_range[0], x_length_range[1])
                    random_y_length = random.uniform(y_length_range[0], y_length_range[1])
                    
                 
                    new_mbr = create_mbr_from_center(
                        y_center=y_center,
                        x_center=x_center,
                        y_length=random_y_length,
                        x_length=random_x_length
                    )
                    
                 
                    mbr_list[idx] = (time_slot, new_mbr)
                    replaced_count += 1
    
    if total_count > 0:
        replaced_percentage = (replaced_count / total_count) * 100
    else:
        replaced_percentage = 0
    
    return new_nested_dict  




def euc_distance(point1, point2):
    """
    point: tuple (y, x) 
    """
    y1, x1 = point1
    y2, x2 = point2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance



def point_to_mbr_euclidean_distance(point, mbr):
    """
    Calculate the shortest euclidean distance between a point and an MBR.
    Point is (y_meters, x_meters), MBR is a nested dictionary with four corners.
    Returns the distance in meters.
    """
    
    y_top = mbr["top_left"]["y_meters"]  
    y_bottom = mbr["bottom_left"]["y_meters"]  
    x_left = mbr["top_left"]["x_meters"]  
    x_right = mbr["top_right"]["x_meters"]  

    p_y, p_x = point

    if y_bottom <= p_y <= y_top and x_left <= p_x <= x_right:
        return 0

    distances = []

    
    if p_x < x_left:
        if y_bottom <= p_y <= y_top:  
            distances.append(euc_distance((p_y, p_x), (p_y, x_left)))
        else:  
            distances.append(euc_distance((p_y, p_x), (y_top, x_left)))
            distances.append(euc_distance((p_y, p_x), (y_bottom, x_left)))


    if p_x > x_right:
        if y_bottom <= p_y <= y_top:  
            distances.append(euc_distance((p_y, p_x), (p_y, x_right)))
        else:  
            distances.append(euc_distance((p_y, p_x), (y_top, x_right)))
            distances.append(euc_distance((p_y, p_x), (y_bottom, x_right)))

 
    if p_y > y_top:
        if x_left <= p_x <= x_right:  
            distances.append(euc_distance((p_y, p_x), (y_top, p_x)))
        else:  
            distances.append(euc_distance((p_y, p_x), (y_top, x_left)))
            distances.append(euc_distance((p_y, p_x), (y_top, x_right)))

    if p_y < y_bottom:
        if x_left <= p_x <= x_right:  
            distances.append(euc_distance((p_y, p_x), (y_bottom, p_x)))
        else:  
            distances.append(euc_distance((p_y, p_x), (y_bottom, x_left)))
            distances.append(euc_distance((p_y, p_x), (y_bottom, x_right)))

    return min(distances)




def merged_mbr_area(mbr1, mbr2):
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
    enclosing_mbr_area = calculate_mbr_area(enclosing_mbr)
    return enclosing_mbr_area


def mbr_to_mbr_euclidean_distance(mbr_a, mbr_b):
    """
    Calculate the shortest distance between two MBRs.
    Each MBR is a nested dictionary with keys: 'top_left', 'top_right', 'bottom_left', 'bottom_right'.
    Returns the shortest distance in meters.
    """
    distances = []


    for _, corner in mbr_a.items():
  
        point = (corner["y_meters"], corner["x_meters"])
    
        dist = point_to_mbr_euclidean_distance(point, mbr_b)
        if dist == 0:
            return 0
        else: 
            distances.append(dist)
    for _, corner in mbr_b.items():
     
        point = (corner["y_meters"], corner["x_meters"])
      
        dist = point_to_mbr_euclidean_distance(point, mbr_a)
        if dist == 0:
            return 0
        else: 
            distances.append(dist)

    return min(distances)




def shortest_distance_between_mbrs_corners(mbr1, mbr2):
    """
    Calculate the shortest distance between points in two dictionaries.

    Parameters:
    dict1 (dict): A dictionary containing y_meters and x_meters pairs in degrees.
    dict2 (dict): Another dictionary containing y_meters and x_meters pairs.

    Returns:
    float: The shortest distance between the points in meters.
    """
    
    if not mbr1 or not mbr2:
        return float('inf')
    
    else:
        shortest_distance = float('inf')
        for key1, point1 in mbr1.items():
            for key2, point2 in mbr2.items():
                location1 = (point1['y_meters'], point1['x_meters'])
                location2 = (point2['y_meters'], point2['x_meters'])
                distance = euc_distance(location1, location2) 
                if distance < shortest_distance:
                    shortest_distance = distance
        return shortest_distance    



