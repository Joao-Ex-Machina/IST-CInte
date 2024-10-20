import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from elitism import eaSimpleWithElitism

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
    
def plot_map(Route):
    # Load the CSV file
    df = pd.read_csv('xy.csv', delimiter=';')
    # Replace commas with periods in Longitude and Latitude columns
    df['Longitude'] = df['Longitude'].str.replace(',', '.').astype(float)
    df['Latitude'] = df['Latitude'].str.replace(',', '.').astype(float)
    # Create a GeoDataFrame with the geometry column
    df_geo = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
    
    # Load the Natural Earth low-res dataset
    world_data = gpd.read_file("geopandas\\ne_110m_admin_0_countries.shp")

    # Get the updated list of unique countries
    countrys = df_geo['Country'].unique()

    # Check for intersection with the 'ADMIN' column for country names
    matched_countries = world_data[world_data['ADMIN'].isin(countrys)]

    # Create the plot
    axis = matched_countries.plot(color='white', edgecolor='black')

    # Plot your GeoDataFrame on top of the map
    df_geo.plot(ax=axis, color='red', markersize=5)

    # Set the limits of the plot to include the UK
    plt.xlim(-25, 45)  # Set the longitude limits
    plt.ylim(30, 85)   # Set the latitude limits

    # Optional: Add titles or labels
    plt.title('Countries in Europe with Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    route_cities = df_geo[df_geo['City'].isin(Route)]
    
    for i in range(len(Route)):
        city_start = route_cities[route_cities['City'] == Route[i]].iloc[0]
        city_end = route_cities[route_cities['City'] == Route[(i+1)%len(Route)]].iloc[0]
        
        # Get the coordinates for the two cities
        start_coords = (city_start['Longitude'], city_start['Latitude'])
        end_coords = (city_end['Longitude'], city_end['Latitude'])
        
        # Create a LineString for the route and plot it
        line = LineString([start_coords, end_coords])
        gpd.GeoSeries([line]).plot(ax=axis, color='blue', linewidth=2)

    # Show the plot
    plt.show()
    
    return
    

#PARAMETERS

RANDOM_SEED = 42
POPULATION_SIZE = 1000
P_CROSSOVER = 0.7
P_MUTATION = 0.2
MAX_GENERATIONS = 250
HALL_OF_FAME_SIZE = 15
random.seed(RANDOM_SEED)


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

toolbox.register("mate", tools.cxOrdered)  # Crossover
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(cost_df))  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", np.mean)
stats.register("min", np.min)

# Create the hall of fame object
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Run the Genetic Algorithm
def main(use_cost=False):
    setup_toolbox(use_cost)  # Set up the toolbox with the selected matrix
    
    # population = toolbox.population(n=POPULATION_SIZE)  # Create initial population of 100 individuals
    
    # result_population, logbook = algorithms.eaSimple(
    #     population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, verbose=False)
    
    #Solution with elitism, the best individuals are kept in the hall of fame
    result_population, logbook = eaSimpleWithElitism(
    toolbox.population(n=POPULATION_SIZE),
    toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    stats=stats,
    halloffame=hof,
    verbose=True
)
    
    # Get the best individual
    best_individual = tools.selBest(result_population, 1)[0]
    best_route = [cities[i] for i in best_individual]  # Convert indices to city names

    print(f"Best route: {best_route}")
    # Print the best ever individual
    if(use_cost):
        print(f"Best cost: {best_individual.fitness.values[0]} €")
    else:
        hour = int(best_individual.fitness.values[0] // 60)
        minute = int(best_individual.fitness.values[0] % 60)
        print(f"Best time: {hour} h {minute} min")

    
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.show()

    plot_map(best_route)
    
if __name__ == "__main__":
    # Set use_cost to True if you want to use cost, otherwise it will use time
    main(use_cost=False)  # Change to True to use cost
