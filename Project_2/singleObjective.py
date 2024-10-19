import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms

def convert_time_to_minutes(time_str):
    #print(time_str)  # For debugging purposes

    # Check if the value is NaN using pd.isna() or a string that indicates no route
    if pd.isna(time_str) or time_str == 'N/A':
        return 99999999999999  # Use very big num for no route

    # Split the time string and handle cases where minutes are not specified
    try:
        parts = time_str.split('h')
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        
        return hours * 60 + minutes
    except ValueError:
        return 99999999999999

def convert_cost(cost_str):
    if pd.isna(cost_str) or cost_str == 'N/A':
        return 99999999999999  # Use very big num for no route
    else:
        return float(cost_str)



# Load time and cost data, assuming both have city names as headers and index
time_df = pd.read_csv('timeplane.csv', sep=';', index_col=0, header=0, dtype=str)
cost_df = pd.read_csv('costplane.csv', sep=';', index_col=0, header=0, dtype=str)

if time_df.columns[-1].startswith('Unnamed'):
    time_df = time_df.iloc[:, :-1]

print(time_df)
print(cost_df)

time_df = time_df.map(convert_time_to_minutes)
cost_df = cost_df.map(convert_cost)

# Get the list of cities (both origin and destination cities are now in index and columns)
cities = list(time_df.index)  # This assumes the cities are the same for both time_df and cost_df

# DEAP setup: Define fitness function (minimization problem)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize distance
creator.create("Individual", list, fitness=creator.FitnessMin)



# Function to evaluate the TSP solution (time or cost matrix)
def eval_tsp(individual, matrix):
    total_value = 0
    num_cities = len(individual)

    for i in range(num_cities):
        # Get city names from the individual (which are city indices)
        city1_idx = individual[i]       # Index of the origin city
        city2_idx = individual[(i + 1) % num_cities]  # Index of the destination city (wrap around at the end)

        # Access the value in the matrix using indices directly
        total_value += matrix[city1_idx, city2_idx]

    return (total_value,)# Setup the DEAP toolbox

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(cities)), len(cities))  # Create a random individual
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Setup the DEAP toolbox based on the chosen evaluation matrix
def setup_toolbox(use_cost=False):
    matrix = cost_df if use_cost else time_df  # Choose the matrix based on user input
    toolbox.register("evaluate", eval_tsp, matrix=matrix.values)  # Register the evaluate function

toolbox.register("mate", tools.cxPartialyMatched)  # Crossover
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

# Run the Genetic Algorithm
def main(use_cost=False):
    setup_toolbox(use_cost)  # Set up the toolbox with the selected matrix
    population = toolbox.population(n=1000)  # Create initial population of 100 individuals
    result_population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=0.7, mutpb=0.2, ngen=250, verbose=False)
    
    # Get the best individual
    best_individual = tools.selBest(result_population, 1)[0]
    best_route = [cities[i] for i in best_individual]  # Convert indices to city names

    print(f"Best route: {best_route}")
    print(f"Best distance (time or cost): {best_individual.fitness.values[0]}")

if __name__ == "__main__":
    # Set use_cost to True if you want to use cost, otherwise it will use time
    main(use_cost=False)  # Change to True to use cost
