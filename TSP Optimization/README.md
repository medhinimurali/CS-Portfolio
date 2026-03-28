# Traveling Salesman Problem (TSP) Project

## Overview
This project implements several algorithms for solving the Traveling Salesman Problem (TSP) and evaluates their performance through experiments. The focus is both on building the algorithms from scratch and analyzing how they compare in terms of solution quality and runtime.

## Input Data (Adjacency Matrices)
The project uses adjacency matrices representing distances between cities.

- Each row represents a city  
- Each value represents the distance between two cities  
- Multiple matrices are provided for each problem size  
- Each size includes several random instances to support reliable experiments  

## Implemented Algorithms

### Part 1: Heuristic Methods
- **Nearest Neighbor (NN)**  
  Greedily selects the closest unvisited city  

- **NN + 2-Opt**  
  Improves the NN solution by iteratively swapping edges to reduce total cost  

- **Repeated Random NN (RRNN)**  
  - Selects randomly among the k nearest neighbors  
  - Uses multiple restarts  
  - Applies 2-opt for further improvement  

---

### Part 2: A* Search
- Uses A* to compute optimal solutions  
- State = partial tour  
- Cost = distance traveled so far  
- Heuristic = Minimum Spanning Tree (MST) over unvisited nodes  

---

### Part 3: Local Search & Genetic Algorithms
- **Hill Climbing**  
- **Simulated Annealing**  
- **Genetic Algorithm**
  - PMX crossover  
  - Mutation  
  - Selection based on fitness  

---

## Experimental Setup

The main focus of this project is comparing algorithm performance through experiments.

For each problem size:
- Multiple adjacency matrices are used  
- Each algorithm is run multiple times per matrix  
- Results are aggregated using the **median**  

Metrics collected:
- Tour cost (solution quality)  
- Real runtime (wall-clock time)  
- CPU runtime  
- Number of nodes expanded (for A*)  

---

## Experiments Performed

### Part 1 Experiments
- Tuned **k** (number of candidate neighbors) for RRNN  
- Tuned **number of restarts**  
- Compared NN, NN+2Opt, and RRNN across different sizes  

---

### Part 2 Experiments
- Compared A* with heuristic methods  
- Analyzed runtime vs solution quality  
- Observed scalability as problem size increases  

---

### Part 3 Experiments
- Compared hill climbing, simulated annealing, and genetic algorithms  
- Tracked how solution quality improves over time  
- Evaluated tradeoffs between exploration and convergence  

---

## How to Run

Run experiments:

Part 1:
python p1_experiments.py

Part 2:
python p2_experiments.py

Part 3:
python p3_exp.py

Run on a single matrix:
python p1.py <matrix_file>

Example:
python p1.py 40_random_adj_mat_1.txt

---

## Key Functions
- cost(route, mat): computes total route cost  
- NN(mat, start): nearest neighbor algorithm  
- NN_2_Opt(route, mat): improves route using 2-opt  
- RNN(mat, k, restarts): randomized nearest neighbor  
- astar_search(): optimal search using A*  
- genetic(): genetic algorithm for TSP  

---

## Summary
This project demonstrates how different approaches to TSP perform in practice. While simpler heuristics are fast, more advanced methods improve solution quality at the cost of runtime. The experiments highlight the tradeoffs between efficiency, scalability, and optimality.