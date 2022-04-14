#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sonia martinez
"""

# Please do not distribute or publish solutions to this
# exercise. You are free to use these problems for educational purposes.

#import sys
#sys.path.append("/Library/Python/3.8/site-packages")

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

from collections import deque
#from queue import PriorityQueue
import math

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
  def dfs(xI, xG, n, m, O, visited_set, tmp_action_list):
    #print (xI)
    #print (xG)
    res = []
    if xI == xG:
      return [a for a in tmp_action_list]
    u = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for i in range(len(u)):
      if not collisionCheck(xI,u[i],O):
        next_x  = tuple(map(lambda a, b: a + b, xI, u[i]))
        if next_x[0] >= 0 and next_x[0] < n:
          if next_x[1] >= 0 and next_x[1] < m:
            if next_x not in visited_set:
              tmp_action_list.append(u[i])
              visited_set.add(xI)
              res = dfs(next_x, xG, n, m, O, visited_set, tmp_action_list)
              tmp_action_list.pop()
    return res
    
  tmp_actions = []
  visited = set()
  res_actions = dfs(xI, xG, n, m, O, visited, tmp_actions)
  #path = getPathFromActions(xI, res_actions)
  cost = getCostOfActions(xI, res_actions, O)

  return res_actions, cost, len(visited)







def breadthFirstSearch(xI,xG,n,m,O):
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
  class Node:
    def __init__(self, cur_x=None, prev_to_cur_action=None, prev=None):
      self.x = cur_x
      self.prev_to_cur_action = prev_to_cur_action
      self.prev = prev

  def bfs(xI, xG, n, m, O):
    #tmp_actions = []
    dst_node = None
    res_actions = []
    visited_set = set()
    q = deque()
    q.append(Node(xI))
    visited_set.add(xI)
    #tmp_actions.append([xI])

    while len(q) != 0:
      print("bfs")
      cur_node = q.popleft()
      cur_x = cur_node.x
      print(cur_x)
      u = [(-1, 0), (0, 1), (1, 0), (0, -1)]
      for i in range(len(u)):
        if not collisionCheck(cur_x,u[i],O):
          next_x  = tuple(map(lambda a, b: a + b, cur_x, u[i]))
          if next_x[0] >= 0 and next_x[0] < n:
            if next_x[1] >= 0 and next_x[1] < m:
              if next_x not in visited_set:
                visited_set.add(next_x)
                next_node = Node(next_x, u[i],cur_node)
                if next_x == xG:
                  dst_node = next_node
                  break
                else:
                  print(next_x)
                  q.append(next_node)

    if dst_node is not None:
      itr = dst_node
      while itr.prev is not None:
        print("back tracing")
        print(itr.prev_to_cur_action)
        res_actions.append(itr.prev_to_cur_action)
        itr = itr.prev
      res_actions.reverse()
    else:
      raise Exception("Error: bfs didn't find a solution")
    
    return res_actions, visited_set

  res_actions, visited = bfs(xI, xG, n, m, O)
  cost = getCostOfActions(xI, res_actions, O)

  return res_actions, cost, len(visited)







def DijkstraSearch(xI,xG,n,m,O,cost):
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
  def djk(xI, xG, n, m, O, cost):
    if cost == "westcost":
      cost = stayWestCost
    elif cost == "eastcost":
      cost = stayEastCost
    else:
      raise Exception(f"Error: expect westcost or eastcost, got {cost}")

    dist = dict()
    prev = dict()
    dist[xI] = 0
    prev[xI] = None

    #q = PriorityQueue()
    q = []
    q.append((0, xI))
    
    visited_set = set()
    visited_set.add(xI)

    q_set = set()
    q_set.add(xI)
    count = 0

    while len(q) != 0:
      print("q")
      count = count + 1
      cur_tup = q.pop()
      cur_x = cur_tup[1]
      q_set.remove(cur_x)
      print(cur_tup)
      u = [(-1, 0), (0, 1), (1, 0), (0, -1)]
      for i in range(len(u)):
        print("direction loop")
        next_x  = tuple(map(lambda a, b: a + b, cur_x, u[i]))
        if next_x != prev[cur_x]:
          if not collisionCheck(cur_x,u[i],O):
            if next_x[0] >= 0 and next_x[0] < n:
              if next_x[1] >= 0 and next_x[1] < m:
                print (f"prev {prev[cur_x]}")
                cur_cost = cost(cur_x, [u[i]], O)
                if cur_cost is not None:
                  cur_dist = dist[cur_x] + cur_cost
                else:
                  raise Exception(f"Error: {cost.__name__} returns None")
                if next_x in dist:
                  if cur_dist < dist[next_x]:
                    dist[next_x] = cur_dist
                    prev[next_x] = cur_x
                else:
                    dist[next_x] = cur_dist
                    prev[next_x] = cur_x
                print(f"next: {next_x}")
                if next_x in q_set:
                  for i in range(len(q)):
                    if q[i][1] == next_x:
                      q[i] = (cur_dist, next_x)
                      break
                else:
                  if next_x not in visited_set:
                    visited_set.add(next_x)
                    q_set.add(next_x)
                    q.append((cur_dist, next_x))
                q.sort(key=lambda x:x[0], reverse=True)
                print(q)
      print(f"direction loop done")
    return dist, prev, count

  res_actions = []
  dist, prev, num_visited = djk(xI, xG, n, m, O, cost)
  print("djk done")

  if prev is not None:
    itr = xG
    while itr != xI:
      print(itr)
      action = tuple(map(lambda cur_x, prev_x: cur_x - prev_x, itr, prev[itr]))
      res_actions.append(action)
      itr = prev[itr]
    res_actions.reverse()
  else:
    raise Exception("Error: djk returns None for prev")

  return res_actions, dist[xG], num_visited
  

  



def manhattanHeuristic(state, goal):
  return abs(goal[0] - state[0]) + abs(goal[1] - state[1])

def euclideanHeuristic(state, goal):
  return math.sqrt(pow(goal[0] - state[0], 2) + pow(goal[1] - state[1], 2))

def nullHeuristic(state,goal):
   """
   A heuristic function estimates the cost from the current state to the nearest
   goal.  This heuristic is trivial.
   """
   return 0

def aStarSearch(xI,xG,n,m,O,heuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  """The function uses a function heuristic as an argument. We have used
    the null heuristic here first, you should redefine heuristics as part of
    the homework.
    Your algorithm also needs to return the total cost of the path using
    getCostofActions functions.
    Finally, the algorithm should return the number of visited
    nodes during the search."""
  "*** YOUR CODE HERE ***"
  def aStar(xI, xG, n, m, O, heuristic):
    if heuristic == "manhattan":
      heuristic = manhattanHeuristic
    elif heuristic == "euclidean":
      heuristic = euclideanHeuristic
    else:
      raise Exception(f"Error: expect manhattan or euclidean, got {heuristic}")

    dist = dict()
    prev = dict()
    dist[xI] = 0
    prev[xI] = None

    #q = PriorityQueue()
    q = []
    q.append((0, xI))

    visited_set = set()
    visited_set.add(xI)

    q_set = set()
    q_set.add(xI)
    count = 0

    while len(q) != 0:
      print("q")
      count = count + 1
      cur_tup = q.pop()
      cur_x = cur_tup[1]
      q_set.remove(cur_x)
      print(cur_tup)
      u = [(-1, 0), (0, 1), (1, 0), (0, -1)]
      for i in range(len(u)):
        print("direction loop")
        next_x  = tuple(map(lambda a, b: a + b, cur_x, u[i]))
        if next_x != prev[cur_x]:
          if not collisionCheck(cur_x,u[i],O):
            if next_x[0] >= 0 and next_x[0] < n:
              if next_x[1] >= 0 and next_x[1] < m:
                print (f"prev {prev[cur_x]}")
                cur_cost = getCostOfActions(cur_x, [u[i]], O)
                cur_heuristic = heuristic(cur_x, xG)
                if cur_cost is None:
                  raise Exception(f"Error: getCosOfActions returns None")
                if cur_heuristic is None:
                  raise Exception(f"Error: {heuristic.__name__} returns None")
                cur_dist = dist[cur_x] + cur_cost + cur_heuristic
                if next_x in dist:
                  if cur_dist < dist[next_x]:
                    dist[next_x] = cur_dist
                    prev[next_x] = cur_x
                else:
                    dist[next_x] = cur_dist
                    prev[next_x] = cur_x
                print(f"next: {next_x}")
                if next_x in q_set:
                  for i in range(len(q)):
                    if q[i][1] == next_x:
                      q[i] = (cur_dist, next_x)
                      break
                else:
                  if next_x not in visited_set:
                    visited_set.add(next_x)
                    q_set.add(next_x)
                    q.append((cur_dist, next_x))
                q.sort(key=lambda x:x[0], reverse=True)
                print(q)
      print(f"direction loop done")
    return dist, prev, count

  res_actions = []
  dist, prev, num_visited = aStar(xI, xG, n, m, O, heuristic)
  print("aStar done")

  if prev is not None:
    itr = xG
    while itr != xI:
      print(itr)
      action = tuple(map(lambda cur_x, prev_x: cur_x - prev_x, itr, prev[itr]))
      res_actions.append(action)
      itr = prev[itr]
    res_actions.reverse()
  else:
    raise Exception("Error: aStar returns None for prev")

  return res_actions, dist[xG], num_visited

    
# Plots the path
def showPath(xI,xG,path,n,m,O):
    gridpath = makePath(xI,xG,path,n,m,O)
    fig, ax = plt.subplots(1, 1) # make a figure + axes
    ax.imshow(gridpath) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)

def test_dfs(xI, xG, path, n, m, O):
    actions, cost, visited_count = depthFirstSearch(xI, xG, n, m, O)
    print("dfs test done")
    print(f"cost {cost}")
    print(f"number of node visited: {visited_count}")
    path = getPathFromActions(xI,actions)
    showPath(xI,xG,path,n,m,O)

def test_bfs(xI, xG, path, n, m, O):
    actions, cost, visited_count = breadthFirstSearch(xI, xG, n, m, O)
    print("bfs test done")
    print(f"cost {cost}")
    print(f"number of node visited: {visited_count}")
    path = getPathFromActions(xI,actions)
    showPath(xI,xG,path,n,m,O)

def test_djk_stay_west_cost(xI, xG, path, n, m, O):
    actions, cost, visited_count = DijkstraSearch(xI, xG, n, m, O, "westcost")
    print("djk stay west test done")
    print(f"cost {cost}")
    print(f"number of node visited: {visited_count}")
    path = getPathFromActions(xI, actions)
    showPath(xI,xG,path,n,m,O)

def test_djk_stay_east_cost(xI, xG, path, n, m, O):
    actions, cost, visited_count = DijkstraSearch(xI, xG, n, m, O, "eastcost")
    print("djk stay east test done")
    print(f"cost {cost}")
    print(f"number of node visited: {visited_count}")
    path = getPathFromActions(xI, actions)
    showPath(xI,xG,path,n,m,O)

def test_astar_manhattanHeuristic(xI, xG, path, n, m, O):
    actions, cost, visited_count = aStarSearch(xI, xG, n, m, O, "manhattan")
    print("astar manhattanHeuristic test done")
    print(f"cost {cost}")
    print(f"number of node visited: {visited_count}")
    path = getPathFromActions(xI, actions)
    showPath(xI,xG,path,n,m,O)

def test_astar_euclideanHeuristic(xI, xG, path, n, m, O):
    actions, cost, visited_count = aStarSearch(xI, xG, n, m, O, "euclidean")
    print("astar euclideanHeuristic test done")
    print(f"cost {cost}")
    print(f"number of node visited: {visited_count}")
    path = getPathFromActions(xI, actions)
    showPath(xI,xG,path,n,m,O)

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
    
    #plt.show()

    test_dfs(xI, xG, path, n, m, O)
    test_bfs(xI, xG, path, n, m, O)
    test_djk_stay_west_cost(xI, xG, path, n, m, O)
    test_djk_stay_east_cost(xI, xG, path, n, m, O)
    test_astar_manhattanHeuristic(xI, xG, path, n, m, O)
    test_astar_euclideanHeuristic(xI, xG, path, n, m, O)
    plt.show()