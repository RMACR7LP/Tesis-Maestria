import math
import sys
from posixpath import normcase
import random
import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np
import itertools

def funciones_biyectivas(numero_jugadores, numero_defensas, id_jugador_con_balon):
      atacantes = numero_jugadores- numero_defensas
      iterable = ''
      for atacante in range(2, atacantes + 2):
            if atacante != id_jugador_con_balon: 
                  iterable += str(atacante)
      
      permutaciones = itertools.permutations(iterable, numero_defensas)
      x = np.arange(atacantes+2, numero_jugadores+2)
      funciones = []
      for perm in permutaciones:
            funcion = []
            for i in range(np.size(x)):
                  funcion.append((x[i], int(perm[i])))
            funciones.append(funcion)
      
      return funciones


def Matriz_Rotacion(s):
    norma = np.linalg.norm(s)
    if norma != 0: 
        cos_theta = s[0]/norma
        sen_theta = s[1]/norma
        R = np.array([[cos_theta, -sen_theta],[sen_theta , cos_theta]])
    else: 
        R = np.array([[1,0],[0,1]])
    return R

# vector_1 = np.array([1,-1])
# vector_2 = np.array([0,1])

# angulo = np.arccos(np.dot(vector_1, vector_2)/np.linalg.norm(vector_1))

# Rotación = Matriz_Rotacion(np.array([np.cos(angulo/3), np.sin(angulo/3)]))
# direcciones = []
# for k in range(0,4):
#       direcciones.append(np.matmul(np.linalg.matrix_power(Rotación, k), vector_1))


velocidades = [np.zeros(2) for i in range(0,7)]
print(velocidades)








