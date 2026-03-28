import os
import sys
import time
import random
import signal
import math
import statistics
import numpy as np
import matplotlib.pyplot as plt

from p2_astar import TSPProblem
sys.path.append(os.path.join(os.path.dirname(__file__), "aima-python"))
from search import astar_search

from p3 import HillClimbProblem, cost
from p3 import pop, fitness, pmx, mutate

# -------------------------
# Timeout helper
# -------------------------

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

# -------------------------
# A*
# -------------------------

def run_astar(mat, time_limit=30):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(time_limit)
    try:
        prob = TSPProblem(mat)
        rstart = time.time_ns()
        cpustart = time.process_time_ns()
        sol = astar_search(prob)
        rend = time.time_ns()
        cpuend = time.process_time_ns()
        signal.alarm(0)
        return sol.path_cost, (rend - rstart), (cpuend - cpustart)
    except TimeoutError:
        return None, None, None

# -------------------------
# Hill Climbing
# -------------------------

def run_hill_climbing_with_history(mat, max_steps):
    problem = HillClimbProblem(mat)
    current = problem.initial
    current_cost = -problem.value(current)
    best_cost = current_cost
    history = [best_cost]
    step = 0
    while step < max_steps:
        actions = problem.actions(current)
        best_neighbor = None
        best_neighbor_val = float("-inf")
        for act in actions:
            nb = problem.result(current, act)
            val = problem.value(nb)
            if val > best_neighbor_val:
                best_neighbor_val = val
                best_neighbor = nb
        if best_neighbor_val <= problem.value(current):
            break
        current = best_neighbor
        current_cost = -problem.value(current)
        if current_cost < best_cost:
            best_cost = current_cost
        history.append(best_cost)
        step += 1
    return current, best_cost, history

def run_hc_timed(mat, max_steps):
    rstart = time.time_ns()
    cpustart = time.process_time_ns()
    tour, best_cost, hist = run_hill_climbing_with_history(mat, max_steps)
    rend = time.time_ns()
    cpuend = time.process_time_ns()
    return best_cost, (rend - rstart), (cpuend - cpustart), hist

# -------------------------
# Simulated Annealing
# -------------------------

def run_sim_anneal_with_history(mat, alpha, temp, max_iterations):
    problem = HillClimbProblem(mat)
    current = problem.initial
    current_val = problem.value(current)
    best_cost = -current_val
    history = [best_cost]
    for t in range(max_iterations):
        T = temp * np.exp(-alpha * t)
        if T <= 1e-12:
            break
        actions = problem.actions(current)
        act = random.choice(actions)
        nb = problem.result(current, act)
        nb_val = problem.value(nb)
        delta = nb_val - current_val
        if delta > 0:
            current = nb
            current_val = nb_val
        else:
            p = np.exp(delta / T)
            if random.random() < p:
                current = nb
                current_val = nb_val
        curr_cost = -current_val
        if curr_cost < best_cost:
            best_cost = curr_cost
        history.append(best_cost)
    return current, best_cost, history

def run_sa_timed(mat, alpha, temp, max_iterations):
    rstart = time.time_ns()
    cpustart = time.process_time_ns()
    tour, best_cost, hist = run_sim_anneal_with_history(mat, alpha, temp, max_iterations)
    rend = time.time_ns()
    cpuend = time.process_time_ns()
    return best_cost, (rend - rstart), (cpuend - cpustart), hist

# -------------------------
# Genetic
# -------------------------

def genetic_with_history(mat, mutation, size, num_gen):
    population = pop(mat, size)

    best_cost = float("inf")
    for t in population:
        c = cost(t, mat)
        if c < best_cost:
            best_cost = c
    history = [best_cost]

    g = 0
    while g < num_gen:
        costs = [cost(t, mat) for t in population]
        max_cost = max(costs)
        fitnesses = [max_cost - c for c in costs]
        total = sum(fitnesses)
        if total == 0:
            weights = [1/len(population)] * len(population)
        else:
            weights = [f/total for f in fitnesses]

        new_children = []
        for _ in range(size // 2):
            p1 = random.choices(population, weights=weights, k=1)[0]
            p2 = random.choices(population, weights=weights, k=1)[0]
            child1, child2 = pmx(p1, p2)
            child1 = mutate(child1, mutation)
            child2 = mutate(child2, mutation)
            new_children.append(child1)
            new_children.append(child2)

        population = population + new_children

        scored = [(cost(t, mat), t) for t in population]
        scored.sort()
        population = [t for c, t in scored[:size]]

        for t in population:
            c = cost(t, mat)
            if c < best_cost:
                best_cost = c
        history.append(best_cost)
        g += 1

    return best_cost, history

def run_ga_timed(mat, mutation, size, num_gen):
    rstart = time.time_ns()
    cpustart = time.process_time_ns()
    best_cost, hist = genetic_with_history(mat, mutation, size, num_gen)
    rend = time.time_ns()
    cpuend = time.process_time_ns()
    return best_cost, (rend - rstart), (cpuend - cpustart), hist

# -------------------------
# Plot helpers
# -------------------------

def plot_three(x, a, b, c, title, ylabel, fname, labels):
    plt.figure()
    plt.plot(x, a, marker='o', label=labels[0])
    plt.plot(x, b, marker='s', label=labels[1])
    plt.plot(x, c, marker='^', label=labels[2])
    plt.xlabel("Number of cities")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.show()

def plot_single_curve(x, y, title, xlabel, ylabel, fname):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(fname)
    plt.show()

def plot_history_curve(hist, title, xlabel, ylabel, fname):
    plt.figure()
    xs = list(range(len(hist)))
    plt.plot(xs, hist, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(fname)
    plt.show()

# -------------------------
# Medians over 10 matrices per size
# -------------------------

def median_metrics_for_size(n, hc_steps, sa_alpha, sa_temp, sa_iters, ga_mut, ga_pop, ga_gen):
    hc_costs, hc_real, hc_cpu = [], [], []
    sa_costs, sa_real, sa_cpu = [], [], []
    ga_costs, ga_real, ga_cpu = [], [], []
    astar_costs, astar_real, astar_cpu = [], [], []

    hc_hist_rep = None
    sa_hist_rep = None
    ga_hist_rep = None

    for i in range(10):
        fname = f"{n}_random_adj_mat_{i}.txt"
        mat = np.loadtxt(fname)

        # A* with timeout
        a_cost, a_rt, a_cpu = run_astar(mat, time_limit = 180)
        if a_cost is not None:
            astar_costs.append(a_cost)
            astar_real.append(a_rt)
            astar_cpu.append(a_cpu)

        # Hill Climb
        hc_cost, hc_rt, hc_cpu_t, hc_hist = run_hc_timed(mat, hc_steps)
        hc_costs.append(hc_cost)
        hc_real.append(hc_rt)
        hc_cpu.append(hc_cpu_t)
        if i == 0:
            hc_hist_rep = hc_hist

        # Sim Anneal
        sa_cost, sa_rt, sa_cpu_t, sa_hist = run_sa_timed(mat, sa_alpha, sa_temp, sa_iters)
        sa_costs.append(sa_cost)
        sa_real.append(sa_rt)
        sa_cpu.append(sa_cpu_t)
        if i == 0:
            sa_hist_rep = sa_hist

        # Genetic
        ga_cost, ga_rt, ga_cpu_t, ga_hist = run_ga_timed(mat, ga_mut, ga_pop, ga_gen)
        ga_costs.append(ga_cost)
        ga_real.append(ga_rt)
        ga_cpu.append(ga_cpu_t)
        if i == 0:
            ga_hist_rep = ga_hist

    # only compute astar median if we have data
    astar_cost_med = statistics.median(astar_costs) if astar_costs else None
    astar_real_med = statistics.median(astar_real) if astar_real else None
    astar_cpu_med  = statistics.median(astar_cpu)  if astar_cpu  else None

    return {
        "hc_cost": statistics.median(hc_costs),
        "sa_cost": statistics.median(sa_costs),
        "ga_cost": statistics.median(ga_costs),
        "astar_cost": astar_cost_med,
        "hc_real": statistics.median(hc_real),
        "sa_real": statistics.median(sa_real),
        "ga_real": statistics.median(ga_real),
        "astar_real": astar_real_med,
        "hc_cpu": statistics.median(hc_cpu),
        "sa_cpu": statistics.median(sa_cpu),
        "ga_cpu": statistics.median(ga_cpu),
        "astar_cpu": astar_cpu_med,
        "hc_hist": hc_hist_rep,
        "sa_hist": sa_hist_rep,
        "ga_hist": ga_hist_rep
    }

# -------------------------
# Part 4 runner
# -------------------------

def run_part4(sizes):
    HC_STEPS = 200
    SA_ALPHA = 0.01
    SA_TEMP = 50.0
    SA_ITERS = 2000
    GA_MUT = 0.10
    GA_POP = 50
    GA_GEN = 100

    hc_cost_ratio, sa_cost_ratio, ga_cost_ratio = [], [], []
    hc_real_ratio, sa_real_ratio, ga_real_ratio = [], [], []
    hc_cpu_ratio,  sa_cpu_ratio,  ga_cpu_ratio  = [], [], []
    ratio_sizes = []  # only sizes where A* succeeded

    hc_abs, sa_abs, ga_abs = [], [], []  # absolute costs for all sizes

    rep_hc_hist = None
    rep_sa_hist = None
    rep_ga_hist = None

    for n in sizes:
        print(f"Running size {n}...")
        data = median_metrics_for_size(
            n, HC_STEPS,
            SA_ALPHA, SA_TEMP, SA_ITERS,
            GA_MUT, GA_POP, GA_GEN
        )

        hc_abs.append(data["hc_cost"])
        sa_abs.append(data["sa_cost"])
        ga_abs.append(data["ga_cost"])

        if data["astar_cost"] is not None:
            ratio_sizes.append(n)
            hc_cost_ratio.append(data["hc_cost"] / data["astar_cost"])
            sa_cost_ratio.append(data["sa_cost"] / data["astar_cost"])
            ga_cost_ratio.append(data["ga_cost"] / data["astar_cost"])
            hc_real_ratio.append(data["hc_real"] / data["astar_real"])
            sa_real_ratio.append(data["sa_real"] / data["astar_real"])
            ga_real_ratio.append(data["ga_real"] / data["astar_real"])
            hc_cpu_ratio.append(data["hc_cpu"] / data["astar_cpu"])
            sa_cpu_ratio.append(data["sa_cpu"] / data["astar_cpu"])
            ga_cpu_ratio.append(data["ga_cpu"] / data["astar_cpu"])

        if rep_hc_hist is None:
            rep_hc_hist = data["hc_hist"]
        if rep_sa_hist is None:
            rep_sa_hist = data["sa_hist"]
        if rep_ga_hist is None:
            rep_ga_hist = data["ga_hist"]

        print(f"Done size {n}")

    # Ratio plots (only for sizes where A* ran)
    if ratio_sizes:
        plot_three(ratio_sizes, hc_real_ratio, sa_real_ratio, ga_real_ratio,
                   "Real Runtime / A*", "Runtime Ratio", "p4_runtime_ratio.png",
                   labels=["Hill Climbing", "Sim Annealing", "Genetic"])
        plot_three(ratio_sizes, hc_cpu_ratio, sa_cpu_ratio, ga_cpu_ratio,
                   "CPU Time / A*", "CPU Ratio", "p4_cpu_ratio.png",
                   labels=["Hill Climbing", "Sim Annealing", "Genetic"])
        plot_three(ratio_sizes, hc_cost_ratio, sa_cost_ratio, ga_cost_ratio,
                   "Cost / A*", "Cost Ratio", "p4_cost_ratio.png",
                   labels=["Hill Climbing", "Sim Annealing", "Genetic"])

    # Absolute cost plot for all sizes
    plot_three(sizes, hc_abs, sa_abs, ga_abs,
               "Absolute Cost by Number of Cities", "Cost", "p4_abs_cost.png",
               labels=["Hill Climbing", "Sim Annealing", "Genetic"])

    # History curves
    if rep_hc_hist is not None:
        plot_history_curve(rep_hc_hist, "Hill Climbing: Best Cost vs Iteration",
                           "Iteration", "Best Cost So Far", "p4_hc_history.png")
    if rep_sa_hist is not None:
        plot_history_curve(rep_sa_hist, "Simulated Annealing: Best Cost vs Iteration",
                           "Iteration", "Best Cost So Far", "p4_sa_history.png")
    if rep_ga_hist is not None:
        plot_history_curve(rep_ga_hist, "Genetic Algorithm: Best Cost vs Generation",
                           "Generation", "Best Cost So Far", "p4_ga_history.png")

    # Hyperparameter plots (on first size, first matrix)
    n0 = sizes[-1]
    mat0 = np.loadtxt(f"{n0}_random_adj_mat_0.txt")

    step_vals = [20, 50, 100, 200, 400]
    step_costs = [run_hc_timed(mat0, s)[0] for s in step_vals]
    plot_single_curve(step_vals, step_costs,
                      "Hill Climbing: max_steps vs Final Cost",
                      "max_steps", "Final Cost", "p4_hc_hyperparam.png")

    alpha_vals = [0.001, 0.005, 0.01, 0.05, 0.1]
    alpha_costs = [run_sa_timed(mat0, a, SA_TEMP, SA_ITERS)[0] for a in alpha_vals]
    plot_single_curve(alpha_vals, alpha_costs,
                      "Sim Annealing: alpha vs Final Cost",
                      "alpha (cooling rate)", "Final Cost", "p4_sa_hyperparam.png")

    mut_vals = [0.01, 0.05, 0.10, 0.20, 0.40]
    mut_costs = [run_ga_timed(mat0, m, GA_POP, GA_GEN)[0] for m in mut_vals]
    plot_single_curve(mut_vals, mut_costs,
                      "Genetic: mutation chance vs Final Cost",
                      "mutation chance", "Final Cost", "p4_ga_hyperparam.png")


# -------------------------

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    sizes = [5, 6, 7, 8, 9]

    run_part4(sizes)