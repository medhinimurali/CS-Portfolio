# Ran in the same folder as the matrices  (Project_1_-_matrices)
# made in virtual environment (venv)

import sys
import time 
import numpy as np 
import random
fname = sys.argv[1]
mat = np.loadtxt(fname)

def cost(route,mat): 
    total = 0 
    for i in range(len(route)-1):
        total += mat[route[i]][route[i+1]]
    return total 

def NN(mat, initial):
    current = initial 
    visited = [initial]
    route = [initial]
    c = float("inf")
    city = -1 
    while len(visited) < len(mat):
        for j in range(len(mat[0])):
            if(j not in visited):
                if(mat[current][j] < c):
                    c = mat[current][j]
                    city = j
        route.append(city)
        visited.append(city)
        city = -1 
        c = float("inf")
    route.append(initial)
    return route

def NN_2_OP_help(route,v1,v2):
    new_route = [] 
    for i in range(0,v1+1):
        new_route.append(route[i])
    for j in range(v2,v1,-1):
        new_route.append(route[j])
    for k in range(v2+1,len(route)):
        new_route.append(route[k])
    return new_route

def NN_2_Opt(route,mat):
    existing_route = route[:]
    improved = True
    while improved: 
        best_distance = cost(existing_route,mat)
        improved = False
        for i in range(0,len(existing_route)-2):
            for j in range(i+1,len(existing_route)-1): 
                new_route = NN_2_OP_help(existing_route,i,j)
                new_distance = cost(new_route,mat)
                if(new_distance < best_distance):
                    existing_route = new_route
                    best_distance = new_distance
                    improved = True 
                    break
            
            if improved == True:
                break
        
        if improved == False:
            break
    return existing_route

def RNN(mat,k,num_restarts):
    curr_cost = float("inf")
    end_route = []
    for i in range(num_restarts):
        start = random.randrange(len(mat))
        curr = start 
        visited = [start]
        route = [start]
        while len(visited) < len(mat): 
            unvisited = [] 
            for j in range(len(mat)):
                if j not in visited: 
                    unvisited.append((mat[curr][j],j))
            unvisited.sort()
            k_cities = unvisited[:k]
            index = random.randrange(len(k_cities))
            city = k_cities[index][1]
            visited.append(city)
            route.append(city)
            curr = city 
        route.append(start)
        rcost = cost(route,mat)
        if rcost < curr_cost:
            curr_cost = rcost 
            end_route = route
        
    final_route = NN_2_Opt(end_route,mat)
    return final_route

if __name__ == "__main__": 
    rstart = time.time_ns()
    cpustart = time.process_time_ns()
    
    nn = NN(mat,0)

    rend = time.time_ns()
    cpuend = time.process_time_ns()

    print("NN")
    print("Cost:", cost(nn,mat))
    print("Real runtime:", rend - rstart )
    print("CPU runtime:", cpuend - cpustart )
    print()


    rstart = time.time_ns()
    cpustart = time.process_time_ns()
    
    nn2 = NN_2_Opt(nn,mat)

    rend = time.time_ns()
    cpuend = time.process_time_ns()

    print("NN_2_OPT")
    print("Cost:", cost(nn2,mat))
    print("Real runtime:", rend - rstart )
    print("CPU runtime:", cpuend - cpustart )
    print()

  

    k = 4
    num_restarts = 10
    
    rstart = time.time_ns()
    cpustart = time.process_time_ns()
    
    RRNN = RNN(mat,k,num_restarts)
    
    rend = time.time_ns()
    cpuend = time.process_time_ns()

    print("RRNN")
    print("k:",k, "num_restarts", num_restarts)
    print("Cost:", cost(RRNN,mat))
    print("Real runtime:", rend - rstart )
    print("CPU runtime:", cpuend - cpustart )
    print()



