---
title: "Applied Computational Intelligence Project Report"
author: "David Gao Xia nº 99907 , João Barreiros C. Rodrigues nº 99668"
date: "Oct. 2024"
output: 
  pdf_document: 
    keep_tex: yes
header-includes:
---

# Single-Objective Genetic Algorithm (SOGA)

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
