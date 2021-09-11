"""
You can create any other helper funtions.
Do not modify the given functions
"""


def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    # TODO
    olist = dict()
    n = len(cost)

    olist.update({start_point:0})

    while len(olist) != 0:
        olist = sorted(olist.items(), key=lambda x: x[1])

        current = olist[0][0]
        olist.pop(0)
        olist = dict(olist)
        # Add current node to path
        path.append(current)
        # If current node is a goal, exit
        if current in goals:
            return path
        # Cycle through graph from right to left
        for i in range(n-1,0,-1):
            # If there is a non self-loop edge between current node and i node and it has not been visited yet
            if cost[current][i] != -1 and cost[current][i] != 0 and i not in path:
                # Get heurstic
                h = heuristic[i]
                # Get cost
                g = cost[current][i]
                # Compute total cost
                f = g + h
                # Update in dictionary
                olist.update({i:f})
    return path
   


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    frontier = []
    n = len(cost)
    # Add starting node to frontier
    frontier.append(start_point)
    # Iterate till no more possible explorable nodes
    while len(frontier) != 0:
        curr = frontier.pop()  
        path.append(curr)
        # If current node is a goal, return
        if curr in goals:
                return path
        # Iterate through matrix from right to left
        for i in range(n - 1, 0, -1):
                if cost[curr][i] != -1 and cost[curr][i] != 0 and (i not in path):
                        frontier.append(i)
    return path
