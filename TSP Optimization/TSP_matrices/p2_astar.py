# Ran in the same folder as the matrices and aima-python
# Ran in a aima virtual environment (aima-env)
import os 
import sys 
import time 
sys.path.append(os.path.join(os.path.dirname(__file__),"aima-python"))

from search import astar_search, Problem

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def mst_cost(mat,unvisited):
    if len(unvisited) <= 1:
        return 0
    unvisited_nodes = []
    for i in unvisited:
        row = []
        for j in unvisited: 
            row.append(mat[i][j])
        unvisited_nodes.append(row)
    mat2 = np.array(unvisited_nodes)
    mst = minimum_spanning_tree(mat2)

    return mst.sum()


class TSPProblem(Problem):
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self,mat):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.mat = mat
        self.n = len(mat)
        initial = ()
        self.expanded = 0 
        self.cache = {}
        super().__init__(initial,None)

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        possible = []
        if len(state) == 0: 
            for i in range(self.n):
                possible.append(i)

        elif len(state) < self.n:
            for i in range(self.n):
                if i not in state:
                    possible.append(i)

        elif len(state) == self.n:
            possible.append(state[0])

        return possible
    
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        res = state + (action,)
        return res

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if len(state) != 0:
            if (state[0] == state[-1]) and (len(state) == self.n+1):
                return True 
        return False 

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        if len(state1) == 0:
            return c
        cost = c + self.mat[state1[-1]][action]
        return cost

    def h(self,node):
        
        self.expanded += 1
        unvisited = []
        for i in range(self.n):
            if i not in node.state:
                unvisited.append(i)
        if len(unvisited) == 0 or len(unvisited) == 1:
            return 0
        
        exist = tuple(sorted(unvisited))
        if exist in self.cache:
            return self.cache[exist]
        c = mst_cost(self.mat,unvisited)
        self.cache[exist] = c
        return c 
 
if __name__ == "__main__":
    fname = sys.argv[1]
    mat = np.loadtxt(fname)

    problem = TSPProblem(mat)
    rstart = time.time_ns()
    cpustart = time.process_time_ns()
    solution = astar_search(problem)
    rend = time.time_ns()
    cpuend = time.process_time_ns()
    rtime = rend-rstart
    cputime = cpuend - cpustart

    expanded = problem.expanded

    print("Route:",solution.state)
    print("Length:",len(solution.state))
    print("Cost:", solution.path_cost)
    print("Real time: ", rtime)
    print("Cpu time: ",cputime)
    print("Expanded: ", expanded)
