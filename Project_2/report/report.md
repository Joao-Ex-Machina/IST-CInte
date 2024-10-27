---
title: "Applied Computational Intelligence Project Report"
author: "David Gao Xia nº 99907 , João Barreiros C. Rodrigues nº 99668"
date: "Oct. 2024"
output: 
  pdf_document: 
    keep_tex: yes
header-includes:
---

# Introduction
This project focuses on solving the Traveling Salesman Problem (TSP) using Evolutionary Computation methods. The goal is to optimize travel across 50 major European cities via collective transport (plane, train, bus), minimizing travel costs or travel time while ensuring each city is visited exactly once.

# Data Collection

The cities selected for our dataset are detailed in the xy.csv file. We gathered data for the plane datasets from Google Flights (https://www.google.com/travel/flights?gl=PT&hl=en). For the bus datasets, we utilized BusBud (https://www.busbud.com/pt-pt). Train dataset information was collected from Omio (https://www.omio.com/companies/trains/comboios-de-portugal-y3ksc).

# Single-Objective Genetic Algorithm (SOGA)

We tackle the Traveling Salesman Problem (TSP) by using Evolutionary Computation with Elitism, were we keep the best individuais on the population instead of replacing it with just the offsprings. 

In cases where we need to find the minimal cost of time for a single type of transport, our population individuals are represented as lists of integers, with each number corresponding to a city in the xy.csv file. The search space has a size of #NumberOfCities!. We employed ordered crossover and shuffle indexes for mutation to ensure that each city is visited exactly once. For selection, we utilized a tournament method of size 3.

In cases where all types of transport (plane, bus, train) are utilized, our population individuals are represented by two lists: one list of integers, similar to the case with a single transport type, and another list of strings ["plane", "bus", "train"], indicating the mode of transport between two cities. The search space has a size of (#NumberOfCities! * 3^#NumberOfCities). We employed ordered crossover for the list of integers and uniform crossover for the transport type list. For mutation, we used shuffle indexing for the integer list and selected a random transport type for the other list. For selection, we utilized a tournament method of size 3.

## Optional Heuristic 

## Results

# Multi-Objective Genetic Algorithm (MOGA)

Our MOGA, has the same foundation of our SOGA, using a eaSimple variation with Elitism, with the option to use the same heuristic described before.

To fit the multi-objective problem we had to changed some components of our algorithm most notably:
    - Fitness registry and calculation/evaluation
    - Selection
    - Solution extraction from result population

The Fitness functions now had to support two objectives, both to be minimized. To achieve this we simply return a tuple from the evaluation function composed of both the respective monetary cost and time taken for a solution.

The Selection is now made using the NSGA-II algorithm:

Finally for the solution extraction from the result population we first calculate two points for the final generation: the ideal point (the minima observed in the possble solutions for both objectives) and the centroid (the average of the non-dominated front solutions). By finding the solution that minimizes the vector distance to both points we extract a middle-ground solution that minimizes both monetary cost and time taken.

## Results
