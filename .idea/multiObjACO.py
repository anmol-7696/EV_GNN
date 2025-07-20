import os
#import collections
#import csv
import pandas as pd
import networkx as nx
import numpy as np
#from collections import OrderedDict
#from sklearn.preprocessing import MinMaxScaler
import random

# --------- ACO CORE FUNCTIONS ---------

def combine_costs(time_matrix, vehicle_matrix, w_time, w_cong):
    return w_time * time_matrix + w_cong * vehicle_matrix

def initialize_pheromone(n):
    return np.ones((n, n))

def select_next_city(current_city, unvisited, pheromone, visibility, alpha, beta):
    pher = pheromone[current_city, unvisited] ** alpha
    vis  = visibility[current_city, unvisited] ** beta
    probs = pher * vis
    if probs.sum() == 0:
        return random.choice(unvisited)
    probs = probs / probs.sum()
    return np.random.choice(unvisited, p=probs)

def construct_tour(n, pheromone, visibility, alpha, beta):
    unvisited = list(range(n))
    current = random.choice(unvisited)
    tour = [current]
    unvisited.remove(current)
    while unvisited:
        nxt = select_next_city(current, unvisited, pheromone, visibility, alpha, beta)
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour

def local_pheromone_update(pheromone, tour, rho):
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        pheromone[a][b] *= (1 - rho)
        pheromone[b][a] = pheromone[a][b]

def global_pheromone_update(pheromone, best_tour, best_cost, Q):
    for i in range(len(best_tour)):
        a, b = best_tour[i], best_tour[(i + 1) % len(best_tour)]
        pheromone[a][b] += Q / best_cost
        pheromone[b][a] = pheromone[a][b]

def tour_cost(tour, cost_matrix):
    total = 0.0
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        total += cost_matrix[a][b]
    return total

def tour_length(tour, time_matrix):
    """
    Compute the total travel time of a tour.
    
    Parameters:
        tour (list): List of node indices representing the tour.
        time_matrix (2D np.array): Travel time between each pair of nodes.
    
    Returns:
        float: Total travel time of the tour (including return to start).
    """
    total = 0.0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]  # wrap around to form a cycle
        travel_time = time_matrix[from_city][to_city]
        if not np.isfinite(travel_time):
            return float('inf')  # Invalid path due to missing connection
        total += travel_time
    return total



def ACO_time_only(time_matrix,
                  num_ants=10, num_iterations=100,
                  alpha=1.0, beta=5.0,
                  rho=0.1, Q=100):
    n = time_matrix.shape[0]

    # Compute visibility: inverse of travel time
    with np.errstate(divide='ignore', invalid='ignore'):
        visibility = 1.0 / time_matrix
        visibility[~np.isfinite(visibility)] = 0.0  # set inf or NaN to 0

    pheromone = initialize_pheromone(n)
    best_tour = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        all_tours = []

        for _ in range(num_ants):
            tour = construct_tour(n, pheromone, visibility, alpha, beta)
            all_tours.append(tour)
            local_pheromone_update(pheromone, tour, rho)

        # Evaluate cost using actual travel times
        all_costs = [tour_length(t, time_matrix) for t in all_tours]
        valid_costs = [(t, c) for t, c in zip(all_tours, all_costs) if np.isfinite(c)]

        if not valid_costs:
            continue  # skip this iteration if no valid tour

        it_best_tour, it_best_cost = min(valid_costs, key=lambda x: x[1])

        if it_best_cost < best_cost:
            best_cost = it_best_cost
            best_tour = it_best_tour[:]

        if best_tour is not None:
            global_pheromone_update(pheromone, best_tour, best_cost, Q)

    if best_tour is None:
        print("Warning: No valid tour found!")
        return None, float('inf')

    return best_tour, best_cost


# --------- CSV PROCESSING ---------

def process_static():
    metaData = pd.read_csv('trafficMeta.csv')
    report_folder = 'traffic_feb_june'

    # Build node index from city names (node IDs)
    node_names = sorted(set(metaData['POINT_1_NAME']) | set(metaData['POINT_2_NAME']))
    node_index = {name: idx for idx, name in enumerate(node_names)}
    n = len(node_names)

    # Build node_id -> city name mapping
    node_to_city = {}
    for i, row in metaData.iterrows():
        node_to_city[row['POINT_1_NAME']] = row['POINT_1_CITY']
        node_to_city[row['POINT_2_NAME']] = row['POINT_2_CITY']

    # Initialize time matrix with infinity
    time_matrix = np.full((n, n), np.inf)

    for i, row in metaData.iterrows():
        report_id = str(row['REPORT_ID'])
        distance = row['DISTANCE_IN_METERS']
        src = row['POINT_1_NAME']
        dst = row['POINT_2_NAME']

        file_path = os.path.join(report_folder,  f"trafficData{report_id}.csv")

        if not os.path.exists(file_path):
            print(f"Missing file for REPORT_ID {report_id}: {file_path}")
            continue

        try:
            traffic_df = pd.read_csv(file_path)
            if 'avgSpeed' not in traffic_df.columns:
                print(f"Missing avgSpeed in file {file_path}")
                continue

            avg_speed = traffic_df['avgSpeed'].mean()
            if pd.isna(avg_speed) or avg_speed <= 0:
                travel_time = np.inf
            else:
                distance = distance / 1000
                travel_time = distance / avg_speed  # Time in seconds or minutes depending on units

            s_idx = node_index[src]
            d_idx = node_index[dst]
            time_matrix[s_idx][d_idx] = travel_time
            time_matrix[d_idx][s_idx] = travel_time  # undirected

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    return time_matrix, node_names, node_to_city


# --------- MAIN EXECUTION ---------
if __name__ == "__main__":
    time_matrix, node_names, node_to_city = process_static()

    finite_vals = time_matrix[np.isfinite(time_matrix)]
    if finite_vals.size > 0:
        max_val = np.max(finite_vals)
        time_matrix = np.where(np.isinf(time_matrix), max_val * 2, time_matrix)

    tour, cost = ACO_time_only(time_matrix)
    if tour is not None:
        print("\n")
        print("Best Tour Node IDs:\n", [node_names[i] for i in tour])

        # Map node IDs to city names
        city_names = [node_to_city.get(node_names[i], "Unknown") for i in tour]
        print("\n")
        print("Best Tour City Names:\n", city_names)
        print("\n")
        print("Total Travel Time:", cost, "hours")
    else:
        print("ACO failed to find a valid tour.")
