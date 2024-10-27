import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.lines as mlines
import os
import platform
import argparse
from elitism import eaSimpleWithElitism 

parser = argparse.ArgumentParser(description="Load CSV from a specified directory")
parser.add_argument("directory", type=str, help="The directory containing the CSV file")
args = parser.parse_args()
directory=args.directory

def convert_time_to_minutes(time_str):
    #print(time_str)  # For debugging purposes

    # Check if the value is NaN using pd.isna() or a string that indicates no route
    if pd.isna(time_str) or time_str == 'N/A' or time_str == '0.0':
        return 9999999999999  # Use very big num for no route

    # Split the time string and handle cases where minutes are not specified
    if 'h' in time_str:
        try:
            parts = time_str.split('h')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            
            return hours * 60 + minutes
        except ValueError:
            return 99999999999999
    if '.' in time_str:
        try:
            parts = time_str.split('.')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            
            return hours * 60 + 60 * minutes * 0.01
        except ValueError:
            return 9999999999999999

def convert_cost(cost_str):
    if pd.isna(cost_str) or cost_str == 'N/A' or cost_str == '0.0':
        return 99999999999999999  # Use very big num for no route
    else:
        return float(cost_str)




def plot_map(best_individual,individual):
    # Load the CSV file
    if os.name == 'posix':  # For Unix-like OSs (Joao)
        df = pd.read_csv(f'{directory}/xy.csv', delimiter=',')
    elif os.name == 'nt':  # For Windows (Xia)
        df = pd.read_csv(f'{directory}\\xy.csv', delimiter=',')
    # Create a GeoDataFrame with the geometry column
    df_geo = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
    
    
    # Load the Natural Earth low-res dataset
    if os.name == 'posix':  # For Unix-like OSs (Joao)
        world_data = gpd.read_file("geopandas/ne_110m_admin_0_countries.shp")
    elif os.name == 'nt':  # For Windows (Xia)
        world_data = gpd.read_file("geopandas\\ne_110m_admin_0_countries.shp")

    # Get the updated list of unique countries
    countrys = df_geo['Country'].unique()

    # Check for intersection with the 'ADMIN' column for country names
    matched_countries = world_data[world_data['ADMIN'].isin(countrys)]

    # Create the plot
    # print(matched_countries.geometry)
    axis = matched_countries.plot(color='white', edgecolor='black')

    # Plot your GeoDataFrame on top of the map
    df_geo.plot(ax=axis, color='red', markersize=5)


    plt.xlim(-25, 45)  # Set the longitude limits
    plt.ylim(30, 85)   # Set the latitude limits
    
    # Optional: Add titles or labels
    plt.title('Countries in Europe with Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    if individual:
        Route = best_individual
        Route = [cities[i] for i in Route]
    
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
    else:
        Route, Transport_modes = best_individual
        Route = [cities[i] for i in Route]
        
        transport_color_map = {
        'plane': 'blue',
        'bus': 'green',
        'train': 'red'
    }

        # Dictionary to track if a legend entry for a transport mode has been added
        legend_added = {'plane': False, 'bus': False, 'train': False}

        # Filter the dataframe to include only the cities in the route
        route_cities = df_geo[df_geo['City'].isin(Route)]
        

        # Loop through the route and plot the lines with the corresponding color
        for i in range(len(Route)):
            city_start = route_cities[route_cities['City'] == Route[i]].iloc[0]
            city_end = route_cities[route_cities['City'] == Route[(i+1) % len(Route)]].iloc[0]

            # Get the coordinates for the two cities
            start_coords = (city_start['Longitude'], city_start['Latitude'])
            end_coords = (city_end['Longitude'], city_end['Latitude'])

            # Create a LineString for the route and plot it with the transport color
            transport_mode = Transport_modes[i]
            line = LineString([start_coords, end_coords])
            gpd.GeoSeries([line]).plot(ax=axis, color=transport_color_map[transport_mode], linewidth=2)

            # Add legend entry if not added before
            if not legend_added[transport_mode]:
                legend_added[transport_mode] = True
                mlines.Line2D([], [], color=transport_color_map[transport_mode], linewidth=2, label=transport_mode)

        # Create the legend and add it to the plot
        handles = [mlines.Line2D([], [], color=color, linewidth=2, label=mode)
                for mode, color in transport_color_map.items()]
        plt.legend(handles=handles, title="Transport Modes")
        
        
    # Show the plot
    plt.show()
    
    return
    

#PARAMETERS

RANDOM_SEED = 41
POPULATION_SIZE = 100
P_CROSSOVER = 0.8
P_MUTATION = 0.9
HEURISTIC_SIZE = 0.25
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 15
random.seed(RANDOM_SEED)


# Load time and cost data, assuming both have city names as headers and index

if os.name == 'posix':  # For Unix-like systems (Linux, macOS)
    plane_time_df = pd.read_csv(f'{directory}/timeplane.csv', sep=',', index_col=0, header=0, dtype=str)
    plane_cost_df = pd.read_csv(f'{directory}/costplane.csv', sep=',', index_col=0, header=0, dtype=str)
    
    bus_time_df = pd.read_csv(f'{directory}/timebus.csv', sep=',', index_col=0, header=0, dtype=str)
    bus_cost_df = pd.read_csv(f'{directory}/costbus.csv', sep=',', index_col=0, header=0, dtype=str)
    
    train_time_df = pd.read_csv(f'{directory}/timetrain.csv', sep=',', index_col=0, header=0, dtype=str)
    train_cost_df = pd.read_csv(f'{directory}/costtrain.csv', sep=',', index_col=0, header=0, dtype=str)
elif os.name == 'nt':  # For Windows
    plane_time_df = pd.read_csv(f'{directory}\\timeplane.csv', sep=',', index_col=0, header=0, dtype=str)
    plane_cost_df = pd.read_csv(f'{directory}\\costplane.csv', sep=',', index_col=0, header=0, dtype=str)
    
    bus_time_df = pd.read_csv(f'{directory}\\timebus.csv', sep=',', index_col=0, header=0, dtype=str)
    bus_cost_df = pd.read_csv(f'{directory}\\costbus.csv', sep=',', index_col=0, header=0, dtype=str)
    
    train_time_df = pd.read_csv(f'{directory}\\timetrain.csv', sep=',', index_col=0, header=0, dtype=str)
    train_cost_df = pd.read_csv(f'{directory}\\costtrain.csv', sep=',', index_col=0, header=0, dtype=str)

if bus_time_df.columns[-1].startswith('Unnamed'):
    bus_time_df = bus_time_df.iloc[:, :-1]
    
if plane_time_df.columns[-1].startswith('Unnamed'):
    plane_time_df = plane_time_df.iloc[:, :-1]

if train_time_df.columns[-1].startswith('Unnamed'):
    train_time_df = train_time_df.iloc[:, :-1]

plane_time_df = plane_time_df.map(convert_time_to_minutes)
plane_cost_df = plane_cost_df.map(convert_cost)

bus_time_df = bus_time_df.map(convert_time_to_minutes)
bus_cost_df = bus_cost_df.map(convert_cost)

train_time_df = train_time_df.map(convert_time_to_minutes)
train_cost_df = train_cost_df.map(convert_cost)

# print(plane_time_df)
# print(plane_cost_df)

def calculate_total_time_and_cost(best_route, best_transport_modes, df_plane_time, df_plane_cost, df_bus_time, df_bus_cost, df_train_time, df_train_cost):
    total_time = 0.0  # Use float for time
    total_cost = 0.0  # Use float for cost
    
    # Define dictionaries to map transport modes to the appropriate time and cost DataFrames
    time_dfs = {
        'plane': df_plane_time,
        'bus': df_bus_time,
        'train': df_train_time
    }
    
    cost_dfs = {
        'plane': df_plane_cost,
        'bus': df_bus_cost,
        'train': df_train_cost
    }

    # Iterate through each leg of the route
    for i in range(len(best_route)):
        city1 = best_route[i]
        city2 = best_route[(i + 1) % len(best_route)]
        
        if isinstance(city1, int) and isinstance(city2, int):
            city1 = cities[city1]
            city2 = cities[city2]
        
        transport_mode = best_transport_modes[i]
    
        # Select the appropriate DataFrames for time and cost based on the transport mode
        df_time = time_dfs[transport_mode]
        df_cost = cost_dfs[transport_mode]
        
        # Retrieve time and cost between city1 and city2 from the selected DataFrames
        try:
            time_value = df_time.loc[city1, city2]  # This should be a numeric value
            cost_value = df_cost.loc[city1, city2]  # This should also be numeric

            print(f"From {city1} to {city2}: {transport_mode} - Time: {time_value} min, Cost: {cost_value} €")
            if isinstance(time_value, pd.Series):
                if len(time_value) > 1:
                    time_value = time_value.iloc[0]

            if isinstance(cost_value, pd.Series):
                if len(cost_value) > 1:
                    cost_value = cost_value.iloc[0]

            # Add the time and cost to the totals
            total_time += time_value
            total_cost += cost_value
            
        except KeyError:
            # If there is no route between city1 and city2, return 0, 0
            return 0,0 

    return total_time, total_cost


if os.name == 'posix':  # For Unix-like systems (Linux, macOS)
    df = pd.read_csv(f'{directory}/xy.csv', delimiter=',')
elif os.name == 'nt':  # For Windows
   df = pd.read_csv(f'{directory}\\xy.csv', delimiter=',')

duplicates = df[df['City'].duplicated(keep=False)]  # This will return all duplicate entries

if not duplicates.empty:
    # Remove the second encounter of duplicates (keep the first occurrence)
    df = df.drop_duplicates(subset='City', keep='first').reset_index(drop=True)

# Get the list of cities (both origin and destination cities are now in index and columns)
cities = list(df['City'])  # This assumes the cities are the same for both time_df and cost_df

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

def multi_eval_tsp(individual,plane_matrix, bus_matrix,train_matrix):
    route, transport_modes = individual
    total_time = 0
    for i in range(len(route)):
        city1_idx = route[i]
        city2_idx  = route[(i+1) % len(route)]
        
        City1 = cities[city1_idx]
        City2 = cities[city2_idx]
        
        plane = plane_matrix[city1_idx,city2_idx]
        bus = bus_matrix[city1_idx,city2_idx]
        train = train_matrix[city1_idx,city2_idx]
        
        # # Replace NaN values with a very large number
        if np.isnan(plane):
            plane = 9999999999999
        if np.isnan(bus):
            bus = 9999999999999
        if np.isnan(train):
            train = 9999999999999
        
        if plane < bus and plane < train: 
            total_time += plane
            transport_modes[i] = 'plane'
            
        elif bus < plane and bus < train:
            total_time += bus
            transport_modes[i] = 'bus'
            
            
        elif train < plane and train < bus:
            total_time += train
            transport_modes[i] = 'train'
        else:
            # Fallback for ties or invalid data
            total_time += min(plane, bus, train)
            transport_modes[i] = 'plane' if plane <= min(bus, train) else ('bus' if bus <= train else 'train')
              
    # time , cost = calculate_total_time_and_cost(route, transport_modes,plane_time_df, plane_cost_df, bus_time_df, bus_cost_df, train_time_df, train_cost_df)       
    # print(time,total_time)
    return (total_time,)  # Tuple for DEAP compatibilityy2_idx]

# def create_individual():
#     while True:
#         route = random.sample(list(range(len(cities))), len(cities))
#         transport_modes = [random.choice(['plane', 'bus', 'train']) for _ in range(len(route))]
#         total_time, total_cost = calculate_total_time_and_cost(route, transport_modes,plane_time_df, plane_cost_df, bus_time_df, bus_cost_df, train_time_df, train_cost_df)
#         if total_time < 999999 or total_cost < 999999 :
#             break;
#     return creator.Individual([route, transport_modes])

def create_individual():
 
    route = random.sample(list(range(len(cities))), len(cities))
    transport_modes = [random.choice(['plane', 'bus', 'train']) for _ in range(len(route))]
    
    return creator.Individual([route, transport_modes])

# def create_heuristic_individual():

#     route = heuristics()
#     transport_modes = [random.choice(['plane', 'bus', 'train']) for _ in range(len(route))]
    
#     return creator.Individual([route, transport_modes])

def create_heuristic_individual():
    
    route = heuristics()
    transport_modes = [random.choice(['plane', 'bus', 'train']) for _ in range(len(route))]
     
    return creator.Individual([route, transport_modes])

toolbox = base.Toolbox()

def heuristics():
    if os.name == 'posix':  # For Unix-like systems (Linux, macOS)
        df = pd.read_csv(f'{directory}/xy.csv', delimiter=',')
    elif os.name == 'nt':  # For Windows
        df = pd.read_csv(f'{directory}\\xy.csv', delimiter=',') 

    route = []
    city_list = list(df['City'])
    
    # Check for duplicates in the 'City' column of df_geo
    duplicates = df[df['City'].duplicated(keep=False)]  # This will return all duplicate entries

    if not duplicates.empty:  
        # Remove the second encounter of duplicates (keep the first occurrence)
        df = df.drop_duplicates(subset='City', keep='first').reset_index(drop=True)
    
    for i in range(len(df)):
        route.append([df['City'][i], df['Longitude'][i], df['Latitude'][i]])
        
    middle = (df['Longitude'].max() + df['Longitude'].min()) / 2

    left_part = []
    right_part = []
    for i in range(len(route)):
        if route[i][1] < middle:
            left_part.append(route[i])
        else:
            right_part.append(route[i])
            
    left_part = sorted(left_part, key=lambda x: (-x[2], -x[1]))
    right_part = sorted(right_part, key=lambda x: (x[2], x[1]))
    
    heuristic_route = []
    for i in range(len(left_part)):
        heuristic_route.append(left_part[i])
    for i in range(len(right_part)):
        heuristic_route.append(right_part[i])


    heuristic_route_indices = [city_list.index(city[0]) for city in heuristic_route]

    return heuristic_route_indices
            

def population_tools(individual=False, heuristic=False):
    if heuristic: 
        if individual:
            toolbox.register("indices", random.sample, range(len(cities)), len(cities))  # Create a random individual
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
            toolbox.register("indices", heuristics)  # Create a random individual
            toolbox.register("heuristic_individual", tools.initIterate, creator.Individual, toolbox.indices)
        else:
            toolbox.register("individual", create_individual)
            toolbox.register("heuristic_individual", create_heuristic_individual)

        def create_population(n):
            heuristic_size = int(HEURISTIC_SIZE * n)

            
            heuristic_individual = [toolbox.heuristic_individual() for _ in range(heuristic_size)]
            random_individual = [toolbox.individual() for _ in range(n - len(heuristic_individual))]
            
            population = heuristic_individual + random_individual
            random.shuffle(population)
            
            return population
        
        toolbox.register("population", create_population)    
                   
    else:
        if individual:
            toolbox.register("indices", random.sample, range(len(cities)), len(cities))  # Create a random individual
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        else:  
            toolbox.register("individual", create_individual)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Setup the DEAP toolbox based on the chosen evaluation matrix
def setup_toolbox(use_cost=False, individual=False, transport = 1):
    if individual:
        if transport == 1:
            matrix = plane_cost_df if use_cost else plane_time_df
        elif transport == 2:
            matrix = bus_cost_df if use_cost else bus_time_df       
        elif transport == 3:
            matrix = train_cost_df if use_cost else train_time_df
        toolbox.register("evaluate", eval_tsp, matrix=matrix.values)  # Register the evaluate function
        
        
    else:              
        plane_matrix = plane_cost_df if use_cost else plane_time_df  # Choose the matrix based on user input
        bus_matrix = bus_cost_df if use_cost else bus_time_df  # Choose the matrix based on user input
        train_matrix = train_cost_df if use_cost else train_time_df  # Choose the matrix based on user input
        toolbox.register("evaluate", multi_eval_tsp, plane_matrix=plane_matrix.values, bus_matrix=bus_matrix.values, train_matrix = train_matrix.values )  # Register the evaluate function
 
def crossover(ind1, ind2):
    # Ordered crossover for cities
#    print(ind1[0])
 #   print(ind2[0])
    route1, route2 = tools.cxOrdered(ind1[0], ind2[0])
    # Uniform crossover for transportation modes
    modes1 = [random.choice([m1, m2]) for m1, m2 in zip(ind1[1], ind2[1])]
    modes2 = [random.choice([m1, m2]) for m1, m2 in zip(ind1[1], ind2[1])]
    ind1[:] = [route1, modes1]
    ind2[:] = [route2, modes2]
    return ind1, ind2 

# Mutation: Swap two cities or randomly change the transportation mode
def mutate(individual, indpb=0.2):
    # Mutate the route by swapping two cities
    if random.random() < indpb:
        tools.mutShuffleIndexes(individual[0], indpb)
    
    # Mutate the transportation modes
    for i in range(len(individual[1])):
        if random.random() < indpb:
            individual[1][i] = random.choice(['plane', 'bus', 'train'])
    return individual,

def offspring_setup(individual = False):
    if individual:
        toolbox.register("mate", tools.cxOrdered)  # Crossover
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(plane_cost_df))  # Mutation
        toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
    else:
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate, indpb=1.0/len(plane_cost_df))
        toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", np.mean)
stats.register("min", np.min)

# Create the hall of fame object
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Run the Genetic Algorithm
def main(use_cost=False, individual=False, transport = 1, heuristic = False):   
    population_tools(individual, heuristic)
    setup_toolbox(use_cost,individual,transport)  # Set up the toolbox with the selected matrix
    offspring_setup(individual)
 
    # population = toolbox.population(n=POPULATION_SIZE)  # Create initial population of 100 individuals
    
    # result_population, logbook = algorithms.eaSimple(
    #     population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, verbose=False)
    
    # Solution with elitism, the best individuals are kept in the hall of fame
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



    
    best_individual = tools.selBest(result_population, 1)[0]
    if individual:
        # Get the best individual
        best_route = [cities[i] for i in best_individual]  # Convert indices to city names
        print(f"Best route: {best_route}")
    else:
        best_route, best_transport_modes = best_individual
        best_route = [cities[i] for i in best_route]
        print("Best route:", best_route)
        print("Best transport modes:", best_transport_modes)
        
    # Print the best ever individual
    if(use_cost):
        print(f"Best cost: {best_individual.fitness.values[0]} €")
    else:
        hour = int(best_individual.fitness.values[0] // 60)
        minute = int(best_individual.fitness.values[0] % 60)
        print(f"Best time: {hour} h {minute} min")

    if individual:
        if transport == 1:
            best_transport_modes = ['plane' for _ in range(len(best_route))]
        elif transport == 2:
            best_transport_modes = ['bus' for _ in range(len(best_route))]
        elif transport == 3:
            best_transport_modes = ['train' for _ in range(len(best_route))]
            
    print("Total time:" , calculate_total_time_and_cost(best_route, best_transport_modes, plane_time_df, plane_cost_df, bus_time_df, bus_cost_df, train_time_df, train_cost_df))

    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
 #   total_time, total_cost = calculate_total_time_and_cost(
  #  best_route, best_transport_modes,
   # plane_time_df, bus_time_df, train_time_df, 
    #plane_cost_df, bus_cost_df, train_cost_df
    #)

#    print(f"Total Time: {total_time}")
 #   print(f"Total Cost: {total_cost}")      
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.show()

    plot_map(best_individual,individual)
    
if __name__ == "__main__":
    # Set use_cost to True if you want to use cost, otherwise it will use time
    main(use_cost=True, individual=False, transport = 1, heuristic = True)  # Change the use_cost to True to use cost, Change individual to True to only one type of transport
                                                          # Trasport type 1: plane, 2: bus 3: train
