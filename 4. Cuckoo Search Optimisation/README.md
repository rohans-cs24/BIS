# Cuckoo Search
## Problem statement
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7379495&tag=1  
**Traffic signal optimisation**  
Static traffic signals waste commuter time and fuel by ignoring real-time road conditions, creating the need for an adaptive system that dynamically optimizes green-light durations to minimize intersection congestion.

```py

import numpy as np
import pandas as pd
import random
from scipy.special import gamma

# =====================================================================
# 1. SETUP & REPRODUCIBILITY
# =====================================================================
np.random.seed(42)
random.seed(42)

# =====================================================================
# 2. PROBLEM PARAMETERS & OBJECTIVE FUNCTION
# =====================================================================
NUM_PHASES = 4
MIN_GREEN = 10   # Minimum green time for safety
MAX_GREEN = 60   # Maximum green time to prevent starvation
CYCLE_TIME = 120 # Fixed total cycle time

# Arrival rates (vehicles per second) for each phase
ARRIVAL_RATES = np.array([0.2, 0.15, 0.25, 0.1]) 
# Saturation flow rates (vehicles per second)
SATURATION_FLOW = np.array([1.0, 1.0, 1.0, 1.0])

def objective_function(green_times):
    """
    Calculates the total average vehicle delay using a simplified Webster-like delay model.
    """
    clearance_time = NUM_PHASES * 4 # 4 seconds lost time per phase
    available_green = CYCLE_TIME - clearance_time
    
    sum_green = np.sum(green_times)
    if sum_green == 0:
        return float('inf')
    
    scaled_green = (green_times / sum_green) * available_green
    scaled_green = np.clip(scaled_green, MIN_GREEN, MAX_GREEN)
    
    sum_green = np.sum(scaled_green)
    scaled_green = (scaled_green / sum_green) * available_green
    
    total_delay = 0
    for i in range(NUM_PHASES):
        g = scaled_green[i]
        c = CYCLE_TIME
        lambda_i = ARRIVAL_RATES[i]
        s = SATURATION_FLOW[i]
        
        capacity = s * (g / c)
        degree_of_sat = lambda_i / capacity if capacity > 0 else 1.0
        
        if degree_of_sat >= 1.0:
            delay = 0.5 * c * (1 - g/c)**2 / (1 - min(0.99, lambda_i/s)) + 1000 * (degree_of_sat - 1.0)
        else:
            delay = 0.5 * c * (1 - g/c)**2 / (1 - lambda_i/s)
            
        total_delay += delay * lambda_i
        
    return total_delay

# =====================================================================
# 3. CUCKOO SEARCH ALGORITHM IMPLEMENTATION
# =====================================================================
def levy_flight(Lambda, size):
    """Generates steps from a Levy flight distribution."""
    temp = gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / (gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))
    sigma_u = temp**(1 / Lambda)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, size=size)
    v = np.random.normal(0, sigma_v, size=size)
    step = u / (np.abs(v)**(1 / Lambda))
    return step

def cuckoo_search_traffic(n_nests=15, max_iter=10, pa=0.25):
    nests = np.random.uniform(MIN_GREEN, MAX_GREEN, (n_nests, NUM_PHASES))
    fitness = np.array([objective_function(nest) for nest in nests])
    
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    history = []
    
    print(f"Initial Best Delay: {best_fitness:.2f} seconds | Green Phases: {np.round(best_nest, 1)}")
    print("-" * 80)
    
    for iteration in range(1, max_iter + 1):
        new_nests = nests.copy()
        step_size = levy_flight(1.5, (n_nests, NUM_PHASES))
        
        for i in range(n_nests):
            step = step_size[i] * 0.01 * (nests[i] - best_nest)
            new_nests[i] += np.random.normal(0, 1, NUM_PHASES) * step
            new_nests[i] = np.clip(new_nests[i], MIN_GREEN, MAX_GREEN)
            
            f_new = objective_function(new_nests[i])
            random_partner_idx = np.random.randint(0, n_nests)
            if f_new < fitness[random_partner_idx]:
                nests[random_partner_idx] = new_nests[i]
                fitness[random_partner_idx] = f_new
        
        for i in range(n_nests):
            if np.random.rand() > pa:
                step_size_abandon = np.random.rand() * (nests[np.random.randint(0, n_nests)] - nests[np.random.randint(0, n_nests)])
                nests[i] += step_size_abandon
                nests[i] = np.clip(nests[i], MIN_GREEN, MAX_GREEN)
                fitness[i] = objective_function(nests[i])
                
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()
            
        history.append((iteration, best_fitness, best_nest.copy()))
        print(f"Iteration {iteration:02d} | Best Delay: {best_fitness:.4f}s | Optimal Green Times: {np.round(best_nest, 1)}")
        
    return best_nest, best_fitness, history

# =====================================================================
# 4. RUN OPTIMIZATION
# =====================================================================
best_solution, best_score, logs = cuckoo_search_traffic(n_nests=20, max_iter=10, pa=0.25)

```

<img width="755" height="275" alt="image" src="https://github.com/user-attachments/assets/81ca6e64-3c92-4862-9793-e5aa31f075d2" />
