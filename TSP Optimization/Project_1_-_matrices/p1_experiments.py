import sys
import time 
import numpy as np 
import random 
import statistics 
import matplotlib.pyplot as plt

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

def median_cost(n,k,num_restarts,trials = 10): 
    c = [] 
    for i in range(10): 
        matrix = str(n) + "_random_adj_mat_" + str(i) + ".txt"
        mat = np.loadtxt(matrix)

        trial_c = []
        
        for j in range(trials):
            route = RNN(mat,k,num_restarts)
            trial_c.append(cost(route,mat))
        c.append(statistics.median(trial_c))

    return statistics.median(c)

def best_k():
    size = [5,10,15,20,25,30]
    num_restarts = 30 
    k = list(range(2,5))
    res = {}
    for i in k: 
        c  = []
        for j in size:
            median = median_cost(j,i,num_restarts)
            c.append(median)
        res[i] = statistics.median(c)
    return res

def best_restarts(k): 
    size = [5,10,15,20,25,30]
    restarts = [10,20,30,40,50,60,70,80,90,100]
    res = {} 
    for i in restarts: 
        c = []
        for j in size: 
            median = median_cost(j,k,i)
            c.append(median)
        res[i] = statistics.median(c)
    return res




def real_runtime(bestk,bestr): 
    NN_time = {}
    Two_OP_Time = {}
    RNN_time = {}
    size = [5,10,15,20,25,30]
    for i in size:
        NN_time_size = []
        Two_OP_Time_Size = [] 
        RNN_time_size = []
        for j in range(10):
            matrix = str(i) + "_random_adj_mat_" + str(j) + ".txt"
            mat = np.loadtxt(matrix)
            NNstart = time.time_ns()
            route = NN(mat,0)
            NNend = time.time_ns() 
            NNtime = NNend-NNstart 
            NN_time_size.append(NNtime)

            NN2start = time.time_ns()
            route = NN(mat,0)
            route2 = NN_2_Opt(route,mat)
            NN2end = time.time_ns() 
            NN2time = NN2end-NN2start 
            Two_OP_Time_Size.append(NN2time)

            RNNstart = time.time_ns()
            route3 = RNN(mat,bestk,bestr)
            RNNend = time.time_ns() 
            RNNtime = RNNend-RNNstart 
            RNN_time_size.append(RNNtime)
        
        NN_time[i] = statistics.median(NN_time_size)
        Two_OP_Time[i] = statistics.median(Two_OP_Time_Size)
        RNN_time[i] = statistics.median(RNN_time_size)
    return NN_time, Two_OP_Time, RNN_time


def cpu_time(bestk,bestr): 
    NN_time = {}
    Two_OP_Time = {}
    RNN_time = {}
    size = [5,10,15,20,25,30]
    for i in size:
        NN_time_size = []
        Two_OP_Time_Size = [] 
        RNN_time_size = []
        for j in range(10):
            matrix = str(i) + "_random_adj_mat_" + str(j) + ".txt"
            mat = np.loadtxt(matrix)
            NNstart = time.process_time_ns()
            route = NN(mat,0)
            NNend = time.process_time_ns() 
            NNtime = NNend-NNstart 
            NN_time_size.append(NNtime)

            NN2start = time.process_time_ns()
            route = NN(mat,0)
            route2 = NN_2_Opt(route,mat)
            NN2end = time.process_time_ns() 
            NN2time = NN2end-NN2start 
            Two_OP_Time_Size.append(NN2time)

            RNNstart = time.process_time_ns()
            route3 = RNN(mat,bestk,bestr)
            RNNend = time.process_time_ns() 
            RNNtime = RNNend-RNNstart 
            RNN_time_size.append(RNNtime)
        
        NN_time[i] = statistics.median(NN_time_size)
        Two_OP_Time[i] = statistics.median(Two_OP_Time_Size)
        RNN_time[i] = statistics.median(RNN_time_size)
    return NN_time, Two_OP_Time, RNN_time

def median_cost_all(bestk,bestr):
    NN_cost = {}
    Two_OP_Cost = {}
    RNN_cost = {}
    size = [5,10,15,20,25,30]
    for i in size:
        NN_cost_size = []
        Two_OP_cost_size = [] 
        RNN_cost_size = []
        for j in range(10):
            matrix = str(i) + "_random_adj_mat_" + str(j) + ".txt"
            mat = np.loadtxt(matrix)
            route = NN(mat,0)
            NN_cost_size.append(cost(route,mat))
            route2 = NN_2_Opt(route,mat)
            Two_OP_cost_size.append(cost(route2,mat))
            route3 = RNN(mat,bestk,bestr)
            RNN_cost_size.append(cost(route3,mat))


        NN_cost[i] = statistics.median(NN_cost_size)
        Two_OP_Cost[i] = statistics.median(Two_OP_cost_size)
        RNN_cost[i] = statistics.median(RNN_cost_size)
    return NN_cost, Two_OP_Cost, RNN_cost



def plot(results, title, outfile, xlabel):
    xs = sorted(results.keys())
    ys = [results[x] for x in xs]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Median cost")
    plt.title(title)
    plt.grid(True)
    plt.savefig(outfile)
    plt.show()
    



def make_comparison_data():
    nn_rt, nn2_rt, rrnn_rt = real_runtime()
    nn_cpu, nn2_cpu, rrnn_cpu = cpu_time()
    nn_cost, nn2_cost, rrnn_cost = median_cost_all()
    return (nn_rt, nn2_rt, rrnn_rt,
            nn_cpu, nn2_cpu, rrnn_cpu,
            nn_cost, nn2_cost, rrnn_cost)

def plot_three(sizes, nn_dict, nn2_dict, rrnn_dict, title, ylabel, outfile):
    nn_y = [nn_dict[n] for n in sizes]
    nn2_y = [nn2_dict[n] for n in sizes]
    rrnn_y = [rrnn_dict[n] for n in sizes]

    plt.figure()
    plt.plot(sizes, nn_y, marker="o", label="NN")
    plt.plot(sizes, nn2_y, marker="s", label="NN+2opt")
    plt.plot(sizes, rrnn_y, marker="^", label="RRNN")

    plt.xticks(sizes)
    plt.xlabel("Number of cities")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(outfile)
    plt.show()


if __name__ == "__main__":
    
    # 1) RRNN hyperparameter tuning plots
    kresults = best_k()
    plot(kresults,
         title="RRNN: k vs Median Cost",
         outfile="k_vs_median_cost.png",
         xlabel="k")

    best_k_val = min(kresults, key=kresults.get)
    print("Best k =", best_k_val)

    # ---- Tune restarts using best k ----
    rresults = best_restarts(best_k_val)
    plot(rresults,
         title="RRNN: num_restarts vs Median Cost",
         outfile="restarts_vs_median_cost.png",
         xlabel="num_restarts")

    best_r_val = min(rresults, key=rresults.get)
    print("Best restarts =", best_r_val)
    # 2) Get comparison data (median runtime / cpu / cost)
    nn_rt, nn2_rt, rrnn_rt = real_runtime(best_k_val, best_r_val)
    nn_cpu, nn2_cpu, rrnn_cpu = cpu_time(best_k_val, best_r_val)
    nn_cost, nn2_cost, rrnn_cost = median_cost_all(best_k_val, best_r_val)

    sizes = [5, 10, 15, 20, 25, 30]

    # 3) Three comparison plots
    plot_three(
        sizes, nn_rt, nn2_rt, rrnn_rt,
        title="Median Real Runtime vs Number of Cities",
        ylabel="Median real runtime (ns)",
        outfile="part1_real_runtime.png"
    )

    plot_three(
        sizes, nn_cpu, nn2_cpu, rrnn_cpu,
        title="Median CPU Time vs Number of Cities",
        ylabel="Median CPU time (ns)",
        outfile="part1_cpu_time.png"
    )

    plot_three(
        sizes, nn_cost, nn2_cost, rrnn_cost,
        title="Median Cost vs Number of Cities",
        ylabel="Median cost",
        outfile="part1_cost.png"
    )
