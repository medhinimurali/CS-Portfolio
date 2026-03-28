
import os
import sys
import time
import numpy as np
import statistics
import matplotlib.pyplot as plt
import random
import signal

from p2_astar import TSPProblem
sys.path.append(os.path.join(os.path.dirname(__file__), "aima-python"))
from search import astar_search

from p1_experiments import NN, NN_2_Opt, RNN, cost

# -------------------------
# Timeout helper
# -------------------------

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

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
        return sol.path_cost, (rend - rstart), (cpuend - cpustart), prob.expanded
    except TimeoutError:
        return None, None, None, None

# -------------------------
# Median metrics
# -------------------------

def median_metrics_for_size(n, best_k, best_r):
    nn_costs, nn2_costs, rrnn_costs = [], [], []
    nn_real, nn2_real, rrnn_real = [], [], []
    nn_cpu, nn2_cpu, rrnn_cpu = [], [], []
    astar_costs, astar_real, astar_cpu, astar_expanded = [], [], [], []

    for i in range(10):
        fname = f"{n}_random_adj_mat_{i}.txt"
        mat = np.loadtxt(fname)

        # A*
        astar_cost, astar_rtime, astar_cputime, astar_exp = run_astar(mat)
        if astar_cost is not None:
            astar_costs.append(astar_cost)
            astar_real.append(astar_rtime)
            astar_cpu.append(astar_cputime)
            astar_expanded.append(astar_exp)

        # NN
        t0 = time.time_ns(); c0 = time.process_time_ns()
        r = NN(mat, 0)
        t1 = time.time_ns(); c1 = time.process_time_ns()
        nn_costs.append(cost(r, mat))
        nn_real.append(t1 - t0)
        nn_cpu.append(c1 - c0)

        # NN + 2opt
        t0 = time.time_ns(); c0 = time.process_time_ns()
        r2 = NN_2_Opt(r, mat)
        t1 = time.time_ns(); c1 = time.process_time_ns()
        nn2_costs.append(cost(r2, mat))
        nn2_real.append(t1 - t0)
        nn2_cpu.append(c1 - c0)

        # RRNN
        t0 = time.time_ns(); c0 = time.process_time_ns()
        r3 = RNN(mat, best_k, best_r)
        t1 = time.time_ns(); c1 = time.process_time_ns()
        rrnn_costs.append(cost(r3, mat))
        rrnn_real.append(t1 - t0)
        rrnn_cpu.append(c1 - c0)

    return {
        "nn_cost": statistics.median(nn_costs),
        "nn2_cost": statistics.median(nn2_costs),
        "rrnn_cost": statistics.median(rrnn_costs),
        "astar_cost": statistics.median(astar_costs) if astar_costs else None,

        "nn_real": statistics.median(nn_real),
        "nn2_real": statistics.median(nn2_real),
        "rrnn_real": statistics.median(rrnn_real),
        "astar_real": statistics.median(astar_real) if astar_real else None,

        "nn_cpu": statistics.median(nn_cpu),
        "nn2_cpu": statistics.median(nn2_cpu),
        "rrnn_cpu": statistics.median(rrnn_cpu),
        "astar_cpu": statistics.median(astar_cpu) if astar_cpu else None,

        "astar_expanded": statistics.median(astar_expanded) if astar_expanded else None
    }

# -------------------------
# Main runner
# -------------------------

def run_part2(sizes, best_k, best_r):
    nn_cost_ratio, nn2_cost_ratio, rrnn_cost_ratio = [], [], []
    nn_real_ratio, nn2_real_ratio, rrnn_real_ratio = [], [], []
    nn_cpu_ratio, nn2_cpu_ratio, rrnn_cpu_ratio = [], [], []
    expanded = []
    ratio_sizes = []

    for n in sizes:
        print(f"Running size {n}...")
        data = median_metrics_for_size(n, best_k, best_r)

        if data["astar_cost"] is not None:
            ratio_sizes.append(n)
            nn_cost_ratio.append(data["nn_cost"] / data["astar_cost"])
            nn2_cost_ratio.append(data["nn2_cost"] / data["astar_cost"])
            rrnn_cost_ratio.append(data["rrnn_cost"] / data["astar_cost"])
            nn_real_ratio.append(data["nn_real"] / data["astar_real"])
            nn2_real_ratio.append(data["nn2_real"] / data["astar_real"])
            rrnn_real_ratio.append(data["rrnn_real"] / data["astar_real"])
            nn_cpu_ratio.append(data["nn_cpu"] / data["astar_cpu"])
            nn2_cpu_ratio.append(data["nn2_cpu"] / data["astar_cpu"])
            rrnn_cpu_ratio.append(data["rrnn_cpu"] / data["astar_cpu"])

        if data["astar_expanded"] is not None:
            expanded.append(data["astar_expanded"])

        print(f"Done size {n}")

    def plot_three(x, a, b, c, title, ylabel, fname):
        plt.figure()
        plt.plot(x, a, marker='o', label='NN')
        plt.plot(x, b, marker='s', label='NN+2opt')
        plt.plot(x, c, marker='^', label='RRNN')
        plt.xlabel("Number of cities")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(fname)
        plt.show()

    if ratio_sizes:
        plot_three(ratio_sizes, nn_real_ratio, nn2_real_ratio, rrnn_real_ratio,
                   "Real Runtime / A*", "Runtime Ratio", "p2_runtime_ratio.png")
        plot_three(ratio_sizes, nn_cpu_ratio, nn2_cpu_ratio, rrnn_cpu_ratio,
                   "CPU Time / A*", "CPU Ratio", "p2_cpu_ratio.png")
        plot_three(ratio_sizes, nn_cost_ratio, nn2_cost_ratio, rrnn_cost_ratio,
                   "Cost / A*", "Cost Ratio", "p2_cost_ratio.png")

    if expanded:
        plt.figure()
        plt.plot(ratio_sizes[:len(expanded)], expanded, marker='o')
        plt.xlabel("Number of cities")
        plt.ylabel("Median nodes expanded by A*")
        plt.title("A* Nodes Expanded vs Size")
        plt.grid(True)
        plt.savefig("p2_expanded.png")
        plt.show()

# -------------------------

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    BEST_K = 4
    BEST_R = 10

    sizes = [5, 6, 7, 8, 9, 10]

    run_part2(sizes, BEST_K, BEST_R) 