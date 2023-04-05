#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report,log_loss,accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
df= pd.read_csv(r"Bank_Personal_Loan_Modelling.csv")
df.drop(['ZIP Code','ID'],axis=1,inplace=True)
X= df.drop('Personal Loan',axis=1).values
y = df['Personal Loan'].values
xtr,xtst,ytr,ytst = tts(X,y,test_size=0.25,stratify=y,random_state=42)
sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xtst = sc.transform(xtst)


# In[4]:


import numpy as np
from sklearn.datasets import make_classification

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.yhat = 1/(1+np.exp(-self.z2))
        return self.yhat

    def compute_loss(self, X, y):
        self.yhat = self.forward(X)
        loss = -np.mean(y*np.log(self.yhat) + (1-y)*np.log(1-self.yhat))
        return loss


# In[5]:


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def initialize_population(self, nn):
        population = []
        for i in range(self.population_size):
            individual = {'W1': np.random.randn(nn.input_size, nn.hidden_size),
                          'W2': np.random.randn(nn.hidden_size, nn.output_size)}
            population.append(individual)
        return population

    def mutate(self, individual):
        for key in individual:
            if np.random.rand() < self.mutation_rate:
                individual[key] += np.random.randn(*individual[key].shape) * 0.1
        return individual

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            if np.random.rand() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def evolve(self, nn, X, y):
        population = self.initialize_population(nn)
        for i in range(100): # number of generations
            # Compute fitness of each individual
            print("gen:",i)
            fitness_scores = []
            for individual in population:
                nn.W1 = individual['W1']
                nn.W2 = individual['W2']
                loss = nn.compute_loss(X, y)
                fitness_scores.append(1 / (1 + loss))
            #print(len(population),np.array(fitness_scores)/sum(fitness_scores))
            # Select parents for mating
            parent1_idx = np.random.choice(range(self.population_size), size=self.population_size, replace=False, p=np.array(fitness_scores)/sum(fitness_scores))
            parent2_idx = np.random.choice(range(self.population_size), size=self.population_size, replace=False, p=np.array(fitness_scores)/sum(fitness_scores))
            parents = [(population[parent1_idx[i]], population[parent2_idx[i]]) for i in range(self.population_size)]

            # Create new generation by crossover and mutation
            new_population = []
            for parent_pair in parents:
                child = self.crossover(parent_pair[0], parent_pair[1])
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
                

            # Select the best individual as the final solution
        fitness_scores = []
        for individual in population:
            nn.W1 = individual['W1']
            nn.W2 = individual['W2']
            loss = nn.compute_loss(X, y)
            fitness_scores.append(1 / (1 + loss))
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        nn.W1 = best_individual['W1']
        nn.W2 = best_individual['W2']
        return nn


# In[6]:


nn = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1)
ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1)


# In[7]:


nn = ga.evolve(nn, xtr, ytr)


# In[8]:


y_pred = np.round(nn.forward(xtst))
accuracy = np.mean(y_pred == ytst)
print("Accuracy: ", accuracy)

