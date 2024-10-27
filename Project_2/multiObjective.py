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
from pymoo.indicators.hv import Hypervolume 

parser = argparse.ArgumentParser(description="Load CSV from a specified directory")
parser.add_argument("directory", type=str, help="The directory containing the CSV file")
args = parser.parse_args()
directory=args.directory
def convert_time_to_minutes(time_str):
    #print(time_str)  # For debugging purposes

    # Check if the value is NaN using pd.isna() or a string that indicates no route
    if pd.isna(time_str) or time_str == 'N/A' or time_str == '0.0':
        return 99999999999999  # Use very big num for no route

    # Split the time string and handle cases where minutes are not specified
    # try:
    #     parts = time_str.split('h')
    #     hours = int(parts[0])
    #     minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        
    #     return hours * 60 + minutes
    # except ValueError:
    #     return 99999999999999
    try:
        parts = time_str.split('.')
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        
        return hours * 60 + 60 * minutes * 0.01
    except ValueError:
        return 99999999999999

def convert_cost(cost_str):
    if pd.isna(cost_str) or cost_str == 'N/A' or cost_str == '0.0':
        return 99999999999999  # Use very big num for no route
    else:
        return float(cost_str)


def calculate_total_time_and_cost(best_route, best_transport_modes, 
                                  df_plane_time, df_plane_cost, 
                                  df_bus_time, df_bus_cost, 
                                  df_train_time, df_train_cost):
    total_time = 0
    total_cost = 0
    
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
    

def calculate_total_time_and_cost(best_route, best_transport_modes, 
                                  df_plane_time, df_plane_cost, 
                                  df_bus_time, df_bus_cost, 
                                  df_train_time, df_train_cost):
    total_time = 0
    total_cost = 0
    
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
    for i in range(len(best_route) - 1):
        city1 = best_route[i]
        city2 = best_route[i + 1]
        transport_mode = best_transport_modes[i]
        
        # Select the appropriate DataFrames for time and cost based on the transport mode
        df_time = time_dfs[transport_mode]
        df_cost = cost_dfs[transport_mode]
        
        # Retrieve time and cost between city1 and city2 from the selected DataFrames
        try:
            time_value = df_time.loc[city1, city2]
            cost_value = df_cost.loc[city1, city2]

            # Check for duplicates in total_time before adding
            if isinstance(total_time, pd.Series) and total_time.index.duplicated().any():
                total_time = total_time[~total_time.index.duplicated(keep='first')]
            
            total_time += time_value
            total_cost += cost_value
            
        except KeyError:
            print(f"No data available for route from {city1} to {city2} via {transport_mode}.")
            continue

        # Add the cost and time for the return to the starting city to complete the route
        city1 = best_route[-1]
        city2 = best_route[0]
        transport_mode = best_transport_modes[-1]

        df_time = time_dfs[transport_mode]
        df_cost = cost_dfs[transport_mode]

        try:
            time_value = df_time.loc[city1, city2]
            cost_value = df_cost.loc[city1, city2]

            # Check for duplicates in total_time before adding
            if isinstance(total_time, pd.Series) and total_time.index.duplicated().any():
                total_time = total_time[~total_time.index.duplicated(keep='first')]

            total_time += time_value
            total_cost += cost_value

        except KeyError:
            print(f"No data available for return route from {city1} to {city2} via {transport_mode}.")

        return total_time, total_cost

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
    


def reduced_matrix(train_time, train_cost,plane_time,plane_cost,bus_time,bus_cost,n):
    
    na_counts = train_time.isna().sum(axis=1)
    
    
    sorted_row_indices = na_counts.nsmallest(n).index
    filtered_df = train_time.loc[sorted_row_indices]
    selected_cities = filtered_df.index.tolist()
    
    train_time_reduced = train_time.loc[selected_cities, selected_cities]
    train_cost_reduced = train_cost.loc[selected_cities, selected_cities]
    plane_time_reduced = plane_time.loc[selected_cities, selected_cities]
    plane_cost_reduced = plane_cost.loc[selected_cities, selected_cities]
    bus_time_reduced = bus_time.loc[selected_cities, selected_cities]
    bus_cost_reduced = bus_cost.loc[selected_cities, selected_cities]
    
    return train_time_reduced, train_cost_reduced, plane_time_reduced, plane_cost_reduced, bus_time_reduced, bus_cost_reduced

def select_n_best_rows_and_corresponding_columns(df,df2, n):
    """
    Select the top N rows with the fewest N/A values and create an NxN matrix 
    where columns correspond to the selected rows.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        n (int): The number of top rows/columns to select.

    Returns:
        pd.DataFrame: An NxN DataFrame containing only the selected rows and their corresponding columns.
    """
    # Count the number of N/A values in each row
    na_counts = df.isna().sum(axis=1)
    
    sorted_na_counts = na_counts.sort_values()
    
    # Print the ordered N/A counts per row for reference
    print("Ordered N/A counts per row:")
    print(sorted_na_counts)

    # Sort rows by N/A counts (ascending) and select the top N rows
    sorted_row_indices = na_counts.nsmallest(n).index

    # Filter the DataFrame to keep only the selected rows
    filtered_df = df.loc[sorted_row_indices]
    filtered_df2 = df2.loc[sorted_row_indices]
    # Get the selected city names (row indices)
    selected_cities = filtered_df.index.tolist()

    # Create an NxN DataFrame where both rows and columns correspond to the selected cities
    # Filter the original DataFrame to include only these rows and columns
    reduced_matrix = df.loc[selected_cities, selected_cities]
    reduced_matrix2 = df2.loc[selected_cities, selected_cities]
    return reduced_matrix, reduced_matrix2

#PARAMETERS

RANDOM_SEED = 42
POPULATION_SIZE = 100
P_CROSSOVER = 0.7
P_MUTATION = 0.3
HEURISTIC_SIZE = 1
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 30
random.seed(RANDOM_SEED)
NCITY = 10;

# Best cost: 2123.4 €
# Best time: 64 h 35 min
# P_CROSSOVER = 0.9
# P_MUTATION = 0.4
# HEURISTIC_SIZE = 0.25


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

# print(train_time_df)
# print(train_cost_df)



if NCITY != 50: 
    train_time_df, train_cost_df, plane_time_df, plane_cost_df, bus_time_df, bus_cost_df = reduced_matrix(train_time_df, train_cost_df,plane_time_df,plane_cost_df,bus_time_df,bus_cost_df,NCITY)

plane_time_df = plane_time_df.map(convert_time_to_minutes)
plane_cost_df = plane_cost_df.map(convert_cost)

bus_time_df = bus_time_df.map(convert_time_to_minutes)
bus_cost_df = bus_cost_df.map(convert_cost)

train_time_df = train_time_df.map(convert_time_to_minutes)
train_cost_df = train_cost_df.map(convert_cost)

if os.name == 'posix':  # For Unix-like systems (Linux, macOS)
    df = pd.read_csv(f'{directory}/xy.csv', delimiter=',')
elif os.name == 'nt':  # For Windows
   df = pd.read_csv(f'{directory}\\xy.csv', delimiter=',')

duplicates = df[df['City'].duplicated(keep=False)]  # This will return all duplicate entries

if not duplicates.empty:
    # Remove the second encounter of duplicates (keep the first occurrence)
    df = df.drop_duplicates(subset='City', keep='first').reset_index(drop=True)

if NCITY != 50:
    df = df[df['City'].isin(train_time_df.columns)]
    df.reset_index(drop=True, inplace=True)
  #Get the list of cities (both origin and destination cities are now in index and columns)
cities = list(df['City'])  # This assumes the cities are the same for both time_df and cost_df

# DEAP setup: Define fitness function (minimization problem)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0))  # Minimize distance
creator.create("Individual", list, fitness=creator.FitnessMulti)


def pareto_eval_tsp(individual, plane_matrix_time, bus_matrix_time, train_matrix_time, 
                    plane_matrix_cost, bus_matrix_cost, train_matrix_cost):
    route, transport_modes = individual
    total_time = 0
    total_cost = 0
    
    # Calculate total time and cost
    for i in range(len(route)):
        city1_idx = route[i]
#        print("Le route")
 #       print(route[i])
        city2_idx = route[(i + 1) % len(route)]
        
        # Retrieve time and cost for each transport mode
        plane_time = plane_matrix_time[city1_idx, city2_idx]
        plane_cost = plane_matrix_cost[city1_idx, city2_idx]
        bus_time = bus_matrix_time[city1_idx, city2_idx]
        bus_cost = bus_matrix_cost[city1_idx, city2_idx]
        train_time = train_matrix_time[city1_idx, city2_idx]
        train_cost = train_matrix_cost[city1_idx, city2_idx]
        if np.isnan(plane_time):
            plane_time = 99999999999
        if np.isnan(plane_cost):
            plane_cost = 99999999999
        if np.isnan(bus_time):
            bus_time = 99999999999
        if np.isnan(bus_cost):
            bus_cost = 99999999999
        if np.isnan(train_time):
            train_time = 99999999999
        if np.isnan(train_cost):
            train_cost = 99999999999



        transport_mode = transport_modes[i]

  #      print(f"TMODE: {transport_mode}")
        # Accumulate time and cost based on the chosen transport mode
        if transport_mode == 'plane':
            total_time += plane_time
            total_cost += plane_cost
        elif transport_mode == 'bus':
            total_time += bus_time
            total_cost += bus_cost
        else:  # train
            total_time += train_time
            total_cost += train_cost
            # Return both objectives: time and cost
    #if (total_time < 99999 or total_cost < 9999999):
    #    print("There is a valid solution somewhere")
    return total_time, total_cost

def plot_pareto_front(population, ideal_point, chosen_solution, centroid):
    times = [ind.fitness.values[0] for ind in population]
    costs = [ind.fitness.values[1] for ind in population]
    plt.scatter(costs, times, c='blue', label='Pareto Front')
    plt.scatter(ideal_point[0], ideal_point[1], c='green', label='Ideal Point')
    plt.scatter(centroid[0], centroid[1], c='purple', label='Centroid')
    plt.scatter(chosen_solution.fitness.values[0], chosen_solution.fitness.values[1],
                c='red', label='Chosen Solution')
    plt.ylabel('Total Time')
    plt.xlabel('Total Cost')
    plt.title('Pareto Front with Ideal Point and Chosen Solution')
    plt.legend()
    plt.show()

def non_dominated_sort(population):
    """Sorts a population into non-dominated fronts."""
    front = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []

        for q in population:
            if p is not q:
                if dominates(p.fitness.values, q.fitness.values):
                    p.dominated_solutions.append(q)
                elif dominates(q.fitness.values, p.fitness.values):
                    p.domination_count += 1

        if p.domination_count == 0:
            front[0].append(p)

    i = 0
    while front[i]:
        next_front = []
        for p in front[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    next_front.append(q)
        i += 1
        front.append(next_front)

    return front[:-1]  # Remove the last empty front


def dominates(individual_a, individual_b):
    """Check if individual_a dominates individual_b."""
    return all(x <= y for x, y in zip(individual_a, individual_b)) and any(x < y for x, y in zip(individual_a, individual_b))


def create_individual():
    route = random.sample(list(range(len(cities))), len(cities))
    transport_modes = [random.choice(['plane', 'bus', 'train']) for _ in range(len(route))]
    return creator.Individual([route, transport_modes])

def create_heuristic_individual():
    
    route = heuristics(df)
    transport_modes = [random.choice(['plane', 'bus', 'train']) for _ in range(len(route))]
     
    return creator.Individual([route, transport_modes])

toolbox = base.Toolbox()

def heuristics(df=df):
   

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
def setup_toolbox():
    plane_matrix_time = plane_time_df.values
    bus_matrix_time = bus_time_df.values
    train_matrix_time = train_time_df.values

    plane_matrix_cost = plane_cost_df.values
    bus_matrix_cost = bus_cost_df.values
    train_matrix_cost = train_cost_df.values
    toolbox.register("evaluate", pareto_eval_tsp,plane_matrix_time=plane_matrix_time, bus_matrix_time=bus_matrix_time, train_matrix_time=train_matrix_time,plane_matrix_cost=plane_matrix_cost, bus_matrix_cost=bus_matrix_cost, train_matrix_cost=train_matrix_cost)


def crossover(ind1, ind2):
    # Ordered crossover for cities
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

def offspring_setup():

        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)  # Selection

def calculate_ideal_point(non_dominated):
    objectives = [ind.fitness.values for ind in non_dominated]

# Calculate minimum values for both objectives
    min_time = min(obj[0] for obj in objectives)
    min_cost = min(obj[1] for obj in objectives)

    # Find the individual with the minimum time and its respective cost
    min_time_individual = next((ind for ind in non_dominated if ind.fitness.values[0] == min_time), None)
    min_cost_individual = next((ind for ind in non_dominated if ind.fitness.values[1] == min_cost), None)

    # Get the respective costs and times
    min_time_cost = min_time_individual.fitness.values[1] if min_time_individual else None
    min_cost_time = min_cost_individual.fitness.values[0] if min_cost_individual else None

    # Print minimum objective values with their respective pairs
    print(f"Minimum Time: {min_time} (Cost: {min_time_cost}) - Individual: {min_time_individual}")
    print(f"Minimum Cost: {min_cost} (Time: {min_cost_time}) - Individual: {min_cost_individual}")

    # Ideal point is derived from the minima in the objective dimensions
    ideal_point = (min_time, min_cost)
    return ideal_point


def calculate_centroid(non_dominated): # find the average point in the non-dominated front
    objectives = np.array([ind.fitness.values for ind in non_dominated])
    centroid = np.mean(objectives, axis=0)
    return centroid

def distance(individual, b): #vector distance between a solution and a explicit point
    return np.linalg.norm(np.array(individual.fitness.values) - b)

def select_solution(non_dominated, ideal_point, centroid):
    
    best_individual = None
    smallest_combined_distance = float('inf')
    
    # Loop through each solution in the non-dominated front
    for individual in non_dominated:
        # Calculate distance to ideal point and centroid
        dist_to_ideal = distance(individual, ideal_point)
        dist_to_centroid = distance(individual, centroid)
        
        # average the distances, this ensures a to find a solution close to the I.P.
        # that minimizes both dimensions
        combined_distance = (dist_to_ideal*1.5 + dist_to_centroid*0.5) / 2
        
        if combined_distance < smallest_combined_distance:
            smallest_combined_distance = combined_distance
            best_individual = individual
    
    return best_individual

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis = 0)
stats.register("min", np.min, axis = 0)

# Create the hall of fame object
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Run the Genetic Algorithm
def main(use_cost=False, individual=False, transport = 1, heuristic = False):   
    population_tools(individual, heuristic)
    setup_toolbox()  # Set up the toolbox with the selected matrix
    offspring_setup()
    
    hypervolume_history = []  # To store hypervolume per generation
    ref_point = [1.1 * 999999999, 1.1 * 999999999]  # Reference point should be worse than any feasible solution
    
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
    

    # Calculate hypervolume for each generation
#    for gen in logbook:
 #       metric = Hypervolume(ref_point=ref_point)
  #      pop_fitnesses = np.array([ind.fitness.values for ind in population in gen])
   #     hypervolume_history.append(metric(pop_fitnesses))   
    
    plt.plot(hypervolume_history, color='blue')
    plt.xlabel('Generations')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume Evolution over Generations')
    plt.show()
    non_dominated = tools.sortNondominated(result_population, len(result_population), first_front_only=True)[0]
    ideal_point= calculate_ideal_point(non_dominated)
    centroid= calculate_centroid(non_dominated)
    best_individual = select_solution(non_dominated, ideal_point, centroid)
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
        hour = int(best_individual.fitness.values[0] // 60)
        minute = int(best_individual.fitness.values[0] % 60)
        print(f"Best time: {hour}h {minute}  and best cost {best_individual.fitness.values[1]}€")
    plot_pareto_front(result_population, ideal_point, best_individual, centroid)
    
    
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    
#  total_time, total_cost = calculate_total_time_and_cost(
 #   best_route, best_transport_modes,
  #  plane_time_df, bus_time_df, train_time_df, 
   # plane_cost_df, bus_cost_df, train_cost_df
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
    main(use_cost=True, individual=False, transport = 3, heuristic = True)  # Change the use_cost to True to use cost, Change individual to True to only one type of transport
                                                          # Trasport type 1: plane, 2: bus 3: train
