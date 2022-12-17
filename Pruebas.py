import math
import sys
from posixpath import normcase
import random
import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np

class Jugador_Simulado(ap.Agent):
    def setup(self):
        self.valor = self.id

    def anotacion(self):
        self.record('valor', self.valor)

    def mostrar(self):
        print(self.model.agents.log[2])

class Jugada_modelo(ap.Model):
    def setup(self):
        self.agents = ap.AgentList(self, 3, Jugador_Simulado)

    def step(self):
        self.agents.anotacion()
        self.agents.mostrar()

parameters = {'steps': 3, 'seed': 123}
    
modelo = Jugada_modelo(parameters)
resultado = modelo.run()