# Ant Coloy Optimization
## Problem statement
**0/1 Knapsack**  
The 0/1 Knapsack Problem is an optimization problem where we have a bag with a limited weight capacity and several items with different weights and values. We must select items to get the maximum total value without exceeding the bag's weight capacity. Each item can either be selected completely (1) or not selected at all (0), meaning we cannot take a fraction of an item.  

```py
import numpy as np
import random

# Knapsack problem data
weights = [2, 3, 4, 5, 9]
values = [3, 4, 5, 8, 10]
capacity = 10

# ACO parameters
num_ants = 20
num_iterations = 100
alpha = 1.0       # Pheromone importance
beta = 2.0        # Heuristic importance
rho = 0.5         # Evaporation rate
Q = 100           # Pheromone deposit factor

n = len(weights)

# Initialize pheromone
pheromone = np.ones(n)

# Heuristic information (value/weight)
heuristic = np.array(values) / np.array(weights)


def construct_solution():
    solution = [0] * n
    remaining_capacity = capacity

    for _ in range(n):
        available = [ 
            i for i in range(n)
            if solution[i] == 0 and weights[i] <= remaining_capacity
        ]

        if not available:
            break

        # Calculate selection probabilities
        probabilities = [
            (pheromone[i] ** alpha) *
            (heuristic[i] ** beta)
            for i in available
        ]

        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Select an item
        selected = random.choices(
            available, weights=probabilities, k=1
        )[0]

        solution[selected] = 1
        remaining_capacity -= weights[selected]

    return solution


def fitness(solution):
    total_weight = sum(
        solution[i] * weights[i] for i in range(n)
    )
    total_value = sum(
        solution[i] * values[i] for i in range(n)
    )

    if total_weight > capacity:
        return 0

    return total_value


# Ant Colony Optimization
best_solution = None
best_value = 0

for iteration in range(num_iterations):
    solutions = []

    # Each ant constructs a solution
    for ant in range(num_ants):
        solution = construct_solution()
        value = fitness(solution)

        solutions.append((solution, value))

        if value > best_value:
            best_value = value
            best_solution = solution.copy()

    # Pheromone evaporation
    pheromone *= (1 - rho)

    # Pheromone update
    for solution, value in solutions:
        if value > 0:
            for i in range(n):
                if solution[i] == 1:
                    pheromone[i] += Q * value / best_value

# Display input configuration and final results
total_weight = sum(
    best_solution[i] * weights[i] for i in range(n)
)

print("--- Input Data ---")
print("Weights: ", weights)
print("Values:  ", values)
print("Capacity:", capacity)

print("\n--- Final Result ---")
print("Selected Items:", best_solution)
print("Total Weight:  ", total_weight)
print("Total Value:   ", best_value)
```
<img width="382" height="171" alt="image" src="https://github.com/user-attachments/assets/37c872bb-edbe-4e2e-8c5f-ebe1cdd8827f" />
