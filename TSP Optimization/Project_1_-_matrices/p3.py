import math
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "aima-python"))


from search import hill_climbing, simulated_annealing, exp_schedule, Problem
import random
import numpy as np 

class HillClimbProblem(Problem):
    def __init__(self,mat):
        self.mat = mat 
        self.n = len(mat)
        initial = self.start_tour()
        super().__init__(initial,None)
    
    def start_tour(self):
        tour = [] 
        for i in range(self.n):
            tour.append(i)
        random.shuffle(tour)
        return tuple(tour)
        
    def actions(self,state):
        swap = []
        for i in range(self.n-1):
            for j in range(i+2,self.n):
                swap.append((i,j))
        return swap 
    
    def result(self,state,action):
        post_swap = []
        for i in state:
            post_swap.append(i)
        first = action[0]
        second = action[1]
        post_swap[first+1:second+1] = reversed(post_swap[first+1:second+1])
        return tuple(post_swap)
    
    def value(self,state):
        cost = 0 
        for i in range(self.n-1):
            x = state[i]
            y = state[i+1]
            cost += self.mat[x][y]
        cost += self.mat[state[-1]][state[0]]
        return -cost

def cost(route,mat): 
    total = 0 
    for i in range(len(route)-1):
        total += mat[route[i]][route[i+1]]
    total += mat[route[-1]][route[0]]
    return total 

def pop(mat, size):
    population = [] 
    for i in range(size):
        tour = []
        for j in range(len(mat)):
            tour.append(j)
        random.shuffle(tour)
        population.append(tuple(tour))
    return population


def fitness(mat,population):
    costs = []
    for i in population: 
        costs.append(cost(i,mat))
    fitnesses = [] 
    for j in costs: 
        fitnesses.append(max(costs) - j )
    if sum(fitnesses) == 0:
        return random.choice(population)
    prob = [] 
    for k in fitnesses:
        prob.append(k/sum(fitnesses))
    parent = random.choices(population,weights = prob, k=1)
    return parent[0]

def genetic(mat, mutation, size, num_gen):
    population = pop(mat,size)
    generation = 0
    while generation < num_gen:
        for i in range(size//2):
            p1 = fitness(mat,population)
            p2 = fitness(mat,population)
            child1, child2 = pmx(p1,p2)
            child1 = mutate(child1,mutation)
            child2 = mutate(child2,mutation)
            population.append(child1)
            population.append(child2)

        pop2 = [] 
        for j in population: 
            c = cost(j,mat)
            pop2.append((c,j))
        
        pop2.sort()
        final_pop = [] 
        count = 0

        for k in pop2: 
            if count < size:
                final_pop.append(k[1])
                count += 1
            else: 
                break 
        
        population = final_pop
        generation += 1 
    best_cost = float("inf")
    best_tour = None 

    for l in population: 
        c = cost(l,mat)
        if c < best_cost:
            best_cost = c
            best_tour = l 
    
    return best_tour
        
def pmx(p1,p2):
    cut = random.sample(range(len(p1)),2)
    mi = min(cut)
    ma = max(cut)

    def child_gen(x,y):
        child = [None] * len(p1)
        mapp = {}
    
        for i in range(mi,ma+1):
            child[i] = x[i]
        
        exist = set(x[mi:ma+1])
        x_index = {}
        for i in range(len(x)):
            x_index[x[i]] = i

        for k in range(len(p1)):
            if mi <= k and k <= ma:
                continue
            value = y[k]
            while value in exist:
                value = y[x_index[value]]
            child[k] = value 
        return tuple(child)
    return child_gen(p1,p2), child_gen(p2,p1)

  

def mutate(tour,prob):
    if random.random() < prob: 
        tour2 = []
        for i in tour:
            tour2.append(i)
        
        i1 = random.randint(0,len(tour)-1)
        i2 = random.randint(0,len(tour)-1)

        while i1 == i2:
            i2 = random.randint(0,len(tour)-1)
        
        swap = tour2[i1]
        tour2[i1] = tour2[i2]
        tour2[i2] = swap
        return tuple(tour2)
    return tour
    

if __name__ == "__main__":
    algorithm = sys.argv[1]
    fname = sys.argv[2]
    mat = np.loadtxt(fname)
    if algorithm == "climbing":
        num_restarts = num_restarts = int(sys.argv[3])
        cost = float("inf")
        state = None
        rstart = time.time_ns()
        cpustart = time.process_time_ns()

        for i in range(num_restarts):
            problem = HillClimbProblem(mat)
            solution = hill_climbing(problem)
            climbcost = -problem.value(solution) 
            if climbcost < cost:
                cost = climbcost
                state = solution
    
        rend = time.time_ns()
        cpuend = time.process_time_ns()

        rtime = rend - rstart
        cputime = cpuend - cpustart

        tour = []
        for i in state:
            tour.append(i)
        tour.append(state[0])

        print("Route: ", tour)
        print("Cost: ", cost)
        print("Real time: ", rtime)
        print("CPU time: ", cputime)
    
    elif algorithm == "anneal": 
        alpha = float(sys.argv[3])
        temp = float(sys.argv[4])
        max_iterations = int(sys.argv[5])

        problem = HillClimbProblem(mat)
        rstart = time.time_ns()
        cpustart = time.process_time_ns()
        solution = simulated_annealing(problem,exp_schedule(k=temp,lam=alpha,limit=max_iterations))

        rend = time.time_ns()
        cpuend = time.process_time_ns()
        tour = []
        for i in solution: 
            tour.append(i)
        tour.append(solution[0])

        print("Route:", tour)
        print("Cost:", -problem.value(solution))
        print("Real time: ", rend-rstart)
        print("CPU time: ", cpuend - cpustart)
    
    elif algorithm == "genetic":
        mutation = float(sys.argv[3])
        size = int((sys.argv[4]))
        gen = int((sys.argv[5]))

        rstart = time.time_ns()
        cpustart = time.process_time_ns()

        tour = genetic(mat,mutation,size,gen)
        c = cost(tour,mat)

        rend = time.time_ns()
        cpuend = time.process_time_ns()

        fintour = []
        for i in tour: 
            fintour.append(i)
        fintour.append(tour[0])

        print("Route:", tour)
        print("Cost:",c)
        print("Real time: ", rend-rstart)
        print("CPU time: ", cpuend - cpustart)

