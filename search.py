#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sonia martinez
"""

# Please do not distribute or publish solutions to this
# exercise. You are free to use these problems for educational purposes.

import sys
sys.path.append("/Library/Python/3.8/site-packages")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from mazemods import maze
from mazemods import makeMaze
from mazemods import collisionCheck
from mazemods import makePath
from mazemods import getPathFromActions
from mazemods import getCostOfActions
from mazemods import stayWestCost
from mazemods import stayEastCost





def depthFirstSearch(xI,xG,n,m,O):
  """
    Search the deepest nodes in the search tree first.
  
    Your search algorithm needs to return a list of actions
    and a path that reaches the goal.  
    Make sure to implement a graph search algorithm.
    Your algorithm also needs to return the cost of the path. 
    Use the getCostOfActions function to do this.
    Finally, the algorithm should return the number of visited
    nodes in your search.
  
    """
  "*** YOUR CODE HERE ***"
  def dfs(xI, xG, n, m, O, action_list):
    if xI == xG:
      return 1
    u = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for i in range(len(u)):
      if not collisionCheck(xI,u[i],O):
        next_x  = tuple(map(lambda i, j: i + j, xI, u[i]))
        if next_x[0] >= 0 and next_x[0] < n:
          if next_x[1] >= 0 and next_x[1] < m:
            action_list.append(u[i])
            return dfs(next_x, xG, n, m, O, action_list) + 1
    return 0
    
  actions = []
  visited_count = dfs(xI, xG, n, m, O, actions)
  path = getPathFromActions(xI, actions)
  cost = getCostOfActions(xI, actions)

  return actions, cost, visited_count







# def breadthFirstSearch(xI,xG,n,m,O):
"""
  Search the shallowest nodes in the search tree first [p 85].
 
  Your search algorithm needs to return a list of actions
  and a path that reaches the goal. Make sure to implement a graph 
  search algorithm.
  Your algorithm also needs to return the cost of the path. 
  Use the getCostOfActions function to do this.
  Finally, the algorithm should return the number of visited
  nodes in your search.

  """
"*** YOUR CODE HERE ***"





# def DijkstraSearch(xI,xG,n,m,O,cost=westCost):
"""
  Search the nodes with least cost first. 
  
  Your search algorithm needs to return a list of actions
  and a path that reaches the goal. Make sure to implement a graph 
  search algorithm.
  Your algorithm also needs to return the total cost of the path using
  either the stayWestCost or stayEastCost function.
  Finally, the algorithm should return the number of visited
  nodes in your search.
  """
"*** YOUR CODE HERE ***"

def nullHeuristic(state,goal):
   """
   A heuristic function estimates the cost from the current state to the nearest
   goal.  This heuristic is trivial.
   """
   return 0

#def aStarSearch(xI,xG,n,m,O,heuristic=nullHeuristic):
"Search the node that has the lowest combined cost and heuristic first."
"""The function uses a function heuristic as an argument. We have used
  the null heuristic here first, you should redefine heuristics as part of 
  the homework. 
  Your algorithm also needs to return the total cost of the path using
  getCostofActions functions. 
  Finally, the algorithm should return the number of visited
  nodes during the search."""
"*** YOUR CODE HERE ***"
  

    
# Plots the path
def showPath(xI,xG,path,n,m,O):
    gridpath = makePath(xI,xG,path,n,m,O)
    fig, ax = plt.subplots(1, 1) # make a figure + axes
    ax.imshow(gridpath) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)
     
if __name__ == '__main__':
    # Run test using smallMaze.py (loads n,m,O)
    from smallMaze import *
    # from mediumMaze import *  # try these mazes too
    # from bigMaze import *     # try these mazes too
    maze(n,m,O) # prints the maze
    
    # Sample collision check
    x, u = (5,4), (1,0)
    testObs = [[6,6,4,4]]
    collided = collisionCheck(x,u,testObs)
    print('Collision!' if collided else 'No collision!')
    
    # Sample path plotted to goal
    xI = (1,1)
    xG = (20,1)
    actions = [(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(0,1),
               (1,0),(1,0),(1,0),(0,-1),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0)]
    path = getPathFromActions(xI,actions)
    showPath(xI,xG,path,n,m,O)
    
    # Cost of that path with various cost functions
    simplecost = getCostOfActions(xI,actions,O)
    westcost = stayWestCost(xI,actions,O)
    eastcost = stayEastCost(xI,actions,O)
    print('Basic cost was %d, stay west cost was %d, stay east cost was %d' %
          (simplecost,westcost,eastcost))
    
    plt.show()