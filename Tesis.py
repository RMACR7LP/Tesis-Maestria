import math
import sys
from posixpath import normcase
import random
import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np


id_posesion = 3 
compañero_a_pasar = -1

def normalizacion(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def probabilidad_de_gol(x,y):
        h = 4400+100  # largo de la cancha (del medio campo hasta la portería?)
        w = 6800 +100+100 # ancho de la cancha
        b = 732 # largo de la portería
        numerador = b*(h-y)
        denominador = (((w-b-2*x)**2/4 + (h-y)**2)*((w+b-2*x)**2/4 + (h-y)**2))**0.5
        angulo = np.arcsin(numerador/denominador)
        probabilidad = 1/(1+np.exp(4.03 - 2.53*angulo - 0.12*(h-y)/100 - 0.11*(h-y)/100*angulo + 0.0069*(h-y)**2/10000))
        return probabilidad

def seguridad_de_pase(x,y): # este debe ser un número entre 0 y 1 
    return 0.5

class Jugador_Simulado(ap.Agent):  

    def setup(self):
        id = self.id
        global id_posesion
        if id == id_posesion: 
            self.estado = 1
        else: 
            self.estado = 0
        self.record('estado', self.estado)
        
        
        

    def setup_pos_s(self, espacio):
        self.espacio = espacio
        self.neighbors = espacio.neighbors   # .neighbors es un metodo de la clase Space que captura todo los vecinos dentro de una cierta distancia.
        self.pos = espacio.positions[self]   # .positions es una variable de la clase Space, es un diccionario que vincula a cada agente con sus coordenadas en el espacio.
        self.record('posiciones', np.array([self.pos[0], self.pos[1]]))


    def cambio_estado(self): 
        id = self.id #el id comienza a contar desde 2
        pos = self.espacio.positions.values()
        pos = np.array(list(pos)).T
        
        # id == 2, Atacante 1
        # id == 3, Atacante 2
        # id == 4, Atacante 3
        # id == 5, Defensa 1
        # id == 6, Defensa 2

        # Definiremos tener posesion de la pelota como estado = 1, no tenerla como estado = 0
        # y disparar como estado 2.

    ########### Defensa #############    
        if self.estado == 0 and 5<=id:
            estado_nuevo = 0

    ###### Atacante Sin Balón #######
        if self.estado == 0 and 2<=id<=4:
            
            global compañero_a_pasar
            if id == compañero_a_pasar+2:
                estado_nuevo = 1
                id_posesion = id
            else: 
                estado_nuevo = 0      

    ###### Atacante con Balón #######
        if self.estado == 1: 
                       
            #Primero checa la probabilidad de gol desde su posicion y desde la de sus compañeros
            P_gol = np.zeros(self.p.atacantes)
            P_gol_propia = probabilidad_de_gol(self.pos[0], self.pos[1])
            for atacante in range(self.p.atacantes):
                P_gol[atacante] = probabilidad_de_gol(pos[0][atacante], pos[1][atacante])

            # Luego, evalúa si está suficientemente confiado para disparar
            confianza = 0.3
            if P_gol_propia> confianza: 
                confiado = 1
            else: 
                confiado = 0
            
            #A continuación evalúa si hay un compañero con mayor probabilidad de gol
            Mejor_compañero = []
            for i in range(self.p.atacantes):
                if P_gol_propia< P_gol[i]:
                    Mejor_compañero.append(i)

            # Ahora, hay 4 escenarios posibles que dependen de (a) si está confiado para disparar
            # y de (b) si hay un compañero con mayor probabilidad de gol: 
            
            individualidad = self.p.individualidades[id-2]
            # Caso 1: a sí y b sí.  
            if confiado == 1 and len(Mejor_compañero)>0:
                conducir = 0
                compañero_a_pasar = -1
                for compañero in Mejor_compañero:
                    if P_gol_propia*individualidad - seguridad_de_pase(pos[0][compañero], pos[1][compañero])*(1-individualidad)< 0: #este sería un número entre 1 y -1
                        if compañero_a_pasar == -1: 
                            compañero_a_pasar = compañero
                        else:
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + seguridad_de_pase(pos[0][compañero_a_pasar], pos[1][compañero_a_pasar])*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + seguridad_de_pase(pos[0][compañero], pos[1][compañero])*0.5
                            if efectividad_2>efectividad_1:
                                compañero_a_pasar = compañero
                if compañero_a_pasar == -1: 
                    disparar = 1
                elif compañero_a_pasar != -1:
                    disparar = 0


            # Caso 2: a sí y b no.
            elif confiado == 1 and len(Mejor_compañero) == 0:
                disparar = 1
                conducir = 0
                
            # Caso 3: a no y b sí:
            elif confiado == 0 and len(Mejor_compañero)>0:
                disparar = 0
                compañero_a_pasar = -1
                for compañero in Mejor_compañero:
                    if seguridad_de_pase(pos[0][compañero], pos[1][compañero])> individualidad: 
                        if compañero_a_pasar == -1:
                            compañero_a_pasar = compañero
                        else:
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + seguridad_de_pase(pos[0][compañero_a_pasar], pos[1][compañero_a_pasar])*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + seguridad_de_pase(pos[0][compañero], pos[1][compañero])*0.5
                            if efectividad_2>=efectividad_1:
                                compañero_a_pasar = compañero
                if compañero_a_pasar == -1:
                    conducir = 1
                elif compañero_a_pasar != -1:
                    conducir = 0

            # Caso 4: a no y b no: 
            else: 
                disparar = 0
                compañero_a_pasar = -1
                for compañero in range(self.p.atacantes):
                    if pos[1][compañero]>self.pos[1] and seguridad_de_pase(pos[0][compañero], pos[1][compañero])>individualidad:
                        if compañero_a_pasar == -1:
                            compañero_a_pasar = compañero
                        else:
                            efectividad_1 = pos[1][compañero_a_pasar]*0.5 + seguridad_de_pase(pos[0][compañero_a_pasar], pos[1][compañero_a_pasar])*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[i]*0.5 + seguridad_de_pase(pos[0][i], pos[1][i])*0.5
                            if efectividad_2>efectividad_1:
                                compañero_a_pasar = compañero
                if compañero_a_pasar == -1: 
                    conducir = 1
                elif compañero_a_pasar != -1:
                    conducir = 0

            # Y aquí estan las reglas de cambio de estado basado en los cuatro casos mencionados
            
            if conducir == 1:
                estado_nuevo = 1
            
            if disparar == 1: 
                estado_nuevo = 2

            if conducir == 0 and disparar == 0: 
                estado_nuevo = 0
        
        if self.estado == 2:
            estado_nuevo = 2

        self.record('estado', estado_nuevo)
        self.estado = estado_nuevo
        

    def cambio_velocidad(self):
        id = self.id #el id comienza a contar desde 2
        pos = self.espacio.positions.values()
        pos = np.array(list(pos)).T
        self.record('posiciones', np.array([self.pos[0], self.pos[1]]))


        Balon_x = self.model.agents.log[id_posesion-2]['posiciones'][-1][0]
        Balon_y = self.model.agents.log[id_posesion-2]['posiciones'][-1][1]
        
        # id == 2, Atacante 1
        # id == 3, Atacante 2
        # id == 4, Atacante 3
        # id == 5, Defensa 1
        # id == 6, Defensa 2
    
    ############## Atacante con Balon ###########
        if self.estado == 1:
            individualidad = self.p.individualidades[id-2]
            # Primero haremos como que el que tiene la pelota siempre va a la direccion del punto medio de la porteria            
            v_1 = normalizacion(np.array([3500-self.pos[0],4500- self.pos[1]]))*individualidad
            v_2 = np.array([0,1])*(1-individualidad)
            velocidad_nueva = 60*normalizacion(v_1+v_2) # el 60 está como comodin para la magnitud de la velocidad

        if self.estado == 2: 
            velocidad_nueva = np.array([0,0])
           
    ############## Atacante sin Balon ############
        if self.estado == 0 and 2<= id<= 4:
            individualidad = 0.5 # esto controla si el jugador busca recibir pase para centrar o un pase para disparar.
            desbordar = self.p.desbordar
            # El jugador debe ubicar la posición del balón y hay esencialmente dos casos: 
            # 1) Si la bola está al centro, el jugador que acompaña busca estar abierto 
            # para jalar la marca y/o ser una opción de pase.
            #if abs(Balon_x-3450)<= abs(self.pos[0] - 3450): 
            v_1 = normalizacion(np.array([3500-self.pos[0],4500- self.pos[1]]))*individualidad #qué tanto quiere ir a la portería?
            if self.pos[0]<=Balon_x:
                v_2 = normalizacion(np.array([1484-self.pos[0], 4500-self.pos[1]]))*(1-individualidad) # qué tanto se va a abrir para buscar recibir el pase y centrar.
            elif self.pos[0]> Balon_x:
                v_2 = normalizacion(np.array([5516-self.pos[0], 4500-self.pos[1]]))*(1-individualidad)
            #2) Si la bola está a un costado, el jugador simplemente busca ir al centro.
            # elif abs(Balon_x-3450)> abs(self.pos[0] - 3450):
            #     if abs(Balon_x - self.pos[0])< 300 and desbordar== 1:
                    
                    
            #     v_1 = normalizacion(np.array([3450-self.pos[0],4500- self.pos[1]]))
            #     v_2 = 0

            # Y en ambos casos debe mantener en cuenta el offside
            v_3 = 0
            ######### PENDIENTE EL OFFSIDE ###########
            ##########################################
            ##########################################

            velocidad_nueva = 60*normalizacion(v_1+v_2+v_3) # aquí el 600 es un comodin para la velocidad maxima
            


    ############## Defensa ################

        if 5 <= id : 
            Rapidez_max = [6.8,6.2] # Estas son las velocidades máximas en m/s
            if self.estado == 0: 
            # Lejos del area, mantenerse relativamente cerca de la marca 
                v_1 = np.array([0, 1])
        
            # Cerca del area, a presionar
                v_2 = 0
                v_3 = 0
            velocidad_nueva = 60*normalizacion(v_1+v_2+v_3)

        self.espacio.move_by(self, velocidad_nueva)
        self.record('velocidad_nueva', velocidad_nueva)
        
        
         
    # def actualizacion(self): 
    #     estado_nuevo = self.model.agents.log[self.id-2]['estado'][-1]
    #     velocidad_nueva = self.model.agents.log[self.id-2]['velocidad_nueva'][-1]
    #     self.estado = estado_nuevo
    #     self.espacio.move_by(self, velocidad_nueva)

            

class Jugada_modelo(ap.Model):

    def setup(self):
        self.espacio = ap.Space(self, shape=[self.p.size]*self.p.dimension)
        self.agents = ap.AgentList(self, 5, Jugador_Simulado) #creamos una cantidad |population| de agentes.
        self.espacio.add_agents(self.agents, [[2000, 1000],[3450, 650], [4200, 1000], [2300, 2000], [4600, 2000]]) #[100,100], [6900, 100], [6900, 4500], [100, 4500]]) #metemos a los agentes creados en el espacio.
        self.agents.setup_pos_s(self.espacio)
        

    def step(self): 
        if self.model.t > 0:
            self.agents.cambio_velocidad()    
            self.agents.cambio_estado()
            
        

def animacion_individual(m, ax):
    ax.set_title(f"{m.t}")
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 4600)
    ax.set_axis_off()
    pos = m.espacio.positions.values()
    pos = np.array(list(pos)).T 
    ax.plot([2584, 2585, 4416, 4416], [4500, 3950, 3950, 4500], color = "white") # area pequeña
    ax.plot([1484, 1484, 5516, 5516], [4500,2850, 2850, 4500], color = "white")  # area grande
    ax.plot([100, 100, 6900, 6900], [100, 4500, 4500, 100], color = "white")  #contorno del campo
    

    estados = []

    for atacante in range(m.p.atacantes):
        temporal = m.agents.log[atacante]
        if len(temporal)!=0:
            estados.append(temporal['estado']) #Esta es una lista con todos los estados que ha pasado el agente i     
        if len(estados)==0:
            estado = 0 
        else: 
            estado = estados[atacante][-1]
        
        
        if estado == 0: 
            ax.scatter(pos[0][atacante],pos[1][atacante], s=50, c='white', marker = "o")
        elif estado == 1: 
            ax.scatter(pos[0][atacante],pos[1][atacante], s=50, c='red', marker = "o")
        elif estado == 2:
            ax.scatter(pos[0][atacante],pos[1][atacante], s=50, c='blue', marker = "o")

    for defensa in range(m.p.atacantes, 5):
        ax.scatter(pos[0][defensa],pos[1][defensa], s=50, c='black', marker = "o")    
     

def animacion_completa(m, p):
    fig = plt.figure(figsize=(10,7))
    fig.patch.set_facecolor("green")
    ax = fig.add_subplot()
    animation = ap.animate(m(p), fig, ax,  animacion_individual, **{"interval": 100} )
    plt.show()
    animation.save("Simulacion movimiento.gif", "GIF")
    return animation

parameters = {
    'dimension': 2,
    'size': 7000,
    'seed': 123,
    'steps': 60,
    'atacantes': 3, 
    'individualidades': np.random.uniform(0,0.5,3),
    'desbordar': np.random.binomial(1,0.5),
}

animacion_completa(Jugada_modelo, parameters)


modelo = Jugada_modelo(parameters)
resultado = modelo.run()
