import math
import sys
from posixpath import normcase
import random
import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools


class Jugador_Simulado(ap.Agent):  

      def setup(self):
            self.id = self.id-2
            

      def setup_pos_s(self, espacio):
            self.espacio = espacio
            self.pos = espacio.positions[self]   # .positions es una variable de la clase Space, es un diccionario que vincula a cada agente con sus coordenadas en el espacio.
            self.record('posiciones', np.array([self.pos[0], self.pos[1]]))

      def printid(self):
            print("\n "+ str(self.id))

class Jugada_modelo(ap.Model):
      def setup(self):
            self.espacio = ap.Space(self, shape=[7000, 9000])
            self.agents = ap.AgentList(self, 5, Jugador_Simulado) #creamos una cantidad |jugadores| de agentes.
            self.espacio.add_agents(self.agents, [[3450, 650],[2000, 1000], [5200, 1000], [2300, 2000], [4600, 2000]]) #metemos a los agentes creados en el espacio.
            self.agents.setup_pos_s(self.espacio)

      def step(self):
            self.agents.printid()
parameters = {
    'dimension': 2,
    'size': 7000,
    'seed': 123,
    'steps': 60,
    'jugadores': 5,
    'atacantes': 3, 
    'defensas': 2,
    'individualidades': np.array([0.8, 0.8, 0.8]),
    'agresividades': np.array([0.01, 0.01]),
}        
modelo = Jugada_modelo(parameters)
resultado = modelo.run()




