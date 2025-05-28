import math
import pandas as pd
import numpy as np
import torch


# Funzione per calcolare la distanza Haversine tra due coordinate
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Raggio della Terra in km
    # Convertire gradi in radianti
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differenze tra latitudini e longitudini
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formula dell'Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distanza in km
    distance = R * c
    return distance


# Funzione per creare la matrice di adiacenza
def create_adjacency_matrix(coordinates, threshold):
    n = len(coordinates)
    # Inizializza la matrice di adiacenza con valori a zero
    adj_matrix = np.zeros((n, n))

    # Calcola le distanze e popola la matrice
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            distance = haversine(lat1, lon1, lat2, lon2)

            # Se la distanza è inferiore alla soglia, aggiungi l'arco con il peso
            if distance < threshold:
                adj_matrix[i][j] = distance
                adj_matrix[j][i] = distance

    # Normalizzazione: Dividi per la distanza massima per ottenere valori tra 0 e 1
    max_distance = np.max(adj_matrix)
    if max_distance > 0:
        adj_matrix /= max_distance

    return torch.tensor(adj_matrix).to('cuda')

if __name__ == '__main__':
    data = pd.read_csv(r'/mnt/c/Users/Grid/Desktop/PhD/EV/EV_GNN/src/dataset/denmark/trafficMetaData.csv')
    list_of_stations = list()
    for row in data.iterrows():
        p1_lat = row[1]['POINT_1_LAT']
        p1_long = row[1]['POINT_1_LNG']
        p2_lat = row[1]['POINT_2_LAT']
        p2_long = row[1]['POINT_2_LNG']
        p_mean_lat = (p1_lat + p2_lat) / 2.0
        p_mean_lng = (p1_long + p2_long) / 2.0
        list_of_stations.append((p_mean_lat, p_mean_lng))

    print(list_of_stations)