# server/server.py
import numpy as np

def federated_average(weight_list):
    print(f"📡 Aggregating {len(weight_list)} client models...")
    avg_weights = list()
    
    for weights in zip(*weight_list):
        avg_weights.append(np.mean(weights, axis=0))
    
    print("✅ Aggregation complete.")
    return avg_weights
