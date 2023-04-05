#!/usr/bin/env python
# coding: utf-8

# # ACO

# In[ ]:


import random


# In[2]:


class ACO(object):

    def __init__(self, num_ants: int, generations: int, alpha: float, beta: float, evaporation: float,
                 cost_matrix: list, num_nodes: int):

        self.num_ants = num_ants
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.cost_matrix = cost_matrix
        self.num_nodes = num_nodes
    def solve(self):
        pheromone_matrix = self.initialisePheromones()

        least_cost_path = []    
        least_cost = 0         

        least_cost_paths_for_gens = []  
        least_cost_for_gens = []        

        for generation in range(self.generations):

            ant_paths = []  
            path_costs = [] 

            for ant in range(self.num_ants):
                startNode = random.randint(0, self.num_nodes - 1)
                currentNode = startNode
                visitedNodes = []
                visitedNodes.append(startNode)

                for node in range(self.num_nodes - 1):
                    currentNode = self.chooseNode(currentNode, visitedNodes, self.num_nodes, pheromone_matrix)
                    visitedNodes.append(currentNode)

                visitedNodes.append(startNode)
                ant_paths.append(visitedNodes)
                path_costs.append(self.calculatePathCost(visitedNodes, self.cost_matrix))

            pheromone_matrix = self.updatePheromones(pheromone_matrix, ant_paths, path_costs)

            least_cost, least_cost_path = self.chooseBestPath(ant_paths, path_costs)
            least_cost_paths_for_gens.append(least_cost_path)
            least_cost_for_gens.append(least_cost)

            if generation == (self.generations - 1):

                least_cost, least_cost_path = self.chooseBestPath(least_cost_paths_for_gens, least_cost_for_gens)

                for i in range(self.num_nodes+1):
                    least_cost_path[i] += 1

        return least_cost, least_cost_path


# In[ ]:


def initialisePheromones(self):
    pheromone_matrix = [[1 for j in range(self.num_nodes)] for i in range(self.num_nodes)]
    return pheromone_matrix
def updatePheromones(self, pheromone_matrix, ant_paths, path_costs):
    updated_pheromone_matrix = pheromone_matrix.copy()
    grouped_pheromone_matrix = [[0 for j in range(self.num_nodes)] for i in range(self.num_nodes)]

    for ant in range(self.num_ants):
        for node in range(self.num_nodes):
            this_node = ant_paths[ant][node]
            next_node = ant_paths[ant][node + 1]

            grouped_pheromone_matrix[this_node][next_node] += (1 / path_costs[ant])

    for i in range(self.num_nodes):
        for j in range(self.num_nodes):
            updated_pheromone_matrix[i][j] = (1 - self.evaporation) * pheromone_matrix[i][j] + grouped_pheromone_matrix[i][j]

    return updated_pheromone_matrix


# In[3]:


def chooseNode(self, currentNode, visitedNodes, num_nodes, pheromone_matrix):

    probabilities = []
    denominator = 0

    for node in range(num_nodes):
        if node in visitedNodes:
            continue
        else:
            denominator += (pheromone_matrix[currentNode][node] ** self.alpha) * ((1 / self.cost_matrix[currentNode][node]) ** self.beta)

    for node in range(num_nodes):
        if node in visitedNodes:
            continue
        else:
            numerator = (pheromone_matrix[currentNode][node] ** self.alpha) * ((1 / self.cost_matrix[currentNode][node]) ** self.beta)
            p = numerator / denominator
            probabilities.append([node, p])

    chosenNode = self.rouletteSelection(probabilities)
    return chosenNode

def rouletteSelection(self, probabilties):
    num_remaining_nodes = len(probabilties)
    p = random.uniform(0, 1)
    upperBound = 1
    probability_values = []

    for i in range(num_remaining_nodes):
        probability_values.append(probabilties[i][1])

    probability_values.sort(reverse=True)

    for i in range(num_remaining_nodes):
        lowerBound = upperBound - probability_values[i]

        if lowerBound < p <= upperBound:
            for j in range(num_remaining_nodes):
                if probabilties[j][1] == probability_values[i]:
                    return probabilties[j][0]
        else:
            upperBound = lowerBound

    return None

def calculatePathCost(self, visited_nodes_list, cost_matrix):
    path_cost = 0

    for i in range(self.num_nodes):
        node = visited_nodes_list[i]
        next_node = visited_nodes_list[i + 1]

        path_cost += cost_matrix[node][next_node]

    return path_cost

def chooseBestPath(self, ant_paths, path_costs):

    index = path_costs.index(min(path_costs))

    least_cost = path_costs[index]
    least_cost_path = ant_paths[index]

    return least_cost, least_cost_path

