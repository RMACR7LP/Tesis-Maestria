import math
import sys
from posixpath import normcase
import random
import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np
import itertools

####################### Notas a considerar ###########################
"""""
En varios puntos del código, hay consideraciones a tomar ya sea por 
cómo funciona la librería agent.py y sus distintas herramientas o bien, 
por las adaptaciones de medidas de la vida real hacia la simulación. En
esta sección hacemos alusión a estas.

1) Cada agente tiene asociado un identificador (self.id) que por razones
que desconozco comienza a correr desde 2. De modo que la lista de identi-
ficadores es [2,3,4,5,6] en una simulación donde hay 5 agentes, lo cual a
veces puede generar confusión porque algunas cosas comienzan en 0. Las
funciones bajo "Funciones Auxiliares" se adaptaron para que en el código
principal se provean como argumentos los self.id directamente. Sin embargo,
en otras partes del código principal a veces se utilizan cosas como
"self.id - 2" y es precisamente por esta razón.


2) La simulación corre a 10 FPS, es decir 1 segundo de la vida real equivale
a 10 fotogramas de la simulación.


"""""


#################### Funciones Auxiliares ############################

"""""
La siguiente función permite hallar todas las funciones biyectivas
el conjunto de defensas y un subconjunto de los atacantes que no contenga
al jugador con posesión del balón. Si los jugadores son {2,3,4,5,6} y están
divididos en atacantes = {2,3,4} y defensas = {5,6}, y el jugador con 
posesión tiene id = 4, entonces el output de esta función es

                    [[(5, 2), (6, 3)], [(5, 3), (6, 2)]]

"""""

def funciones_biyectivas(numero_jugadores, numero_defensas, posesión):
      atacantes = numero_jugadores- numero_defensas
      iterable = ''
      for atacante in range(2, atacantes+2):
            if atacante != posesión: 
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

def distancia_entre_agentes(id_agente_1, id_agente_2, matriz_de_posiciones): 
    """""
    Nótese que utilizamos la matriz de posiciones del agentpy que corre de 0 al número de jugadores
    tanto en sus filas como en las columnas, y pedimos para esta función los id de los agentes 
    según agentpy que corren de 2 hasta el número de jugadores + 2. Por ello, en las variables a continuación
    aparecen cosas como id_agente_1 - 2.
    """""
    dist_x = matriz_de_posiciones[0][id_agente_1-2] - matriz_de_posiciones[0][id_agente_2-2]
    dist_y = matriz_de_posiciones[0][id_agente_1-2] - matriz_de_posiciones[0][id_agente_2-2]
    return np.sqrt(dist_x**2 + dist_y**2)  

def angulo_entre_agentes(id_agente_1, id_agente_2, matriz_de_posiciones):
    pos_defensa = np.array([matriz_de_posiciones[0][id_agente_1-2], matriz_de_posiciones[1][id_agente_1-2]])
    pos_jugador_con_balon = np.array([matriz_de_posiciones[0][id_agente_2-2], matriz_de_posiciones[1][id_agente_2-2]])
    vector_1 = pos_jugador_con_balon - pos_defensa
    vector_1 = vector_1/np.linalg.norm(vector_1)
    vector_2 = np.array([0,1])
    angulo =  np.arccos(np.dot(vector_1, vector_2))
    if vector_1[0]<0: 
        angulo = - angulo
    return angulo, vector_1

#################### Modelo de Control de Cancha #####################

# Nota: En el modelo de control de cancha las escalas están 1:1
# es decir, R= 10 representa una distancia de 10 metros. Por otro lado,
# en el espacio de la simulacion, la relacion es 100:1. Es decir R = 100
# representa una distancia de 1 metro.


def Radio_Influencia(D):
    if 0 <= D <= 10*np.cbrt(6):
        r = D**3/1000+4
    elif D>10*np.cbrt(6):
        r = 10
    else: 
        r = 0
    return r

def Matriz_Escala(D, s):
    Radio = Radio_Influencia(D)
    S_rat = np.linalg.norm(s)**2/13**2
    S = np.array([[(Radio+Radio*S_rat)/2, 0], [0, (Radio-Radio*S_rat)/2]])
    return S

def Matriz_Rotacion(s):
    norma = np.linalg.norm(s)
    if norma != 0: 
        cos_theta = s[0]/norma
        sen_theta = s[1]/norma
        R = np.array([[cos_theta, -sen_theta],[sen_theta , cos_theta]])
    else: 
        R = np.array([[1,0],[0,1]])
    return R

def Covarianza(D, s):
    R = Matriz_Rotacion(s)
    S = Matriz_Escala(D,s)
    R_inv = np.linalg.inv(R)
    COV = np.matmul(R,np.matmul(S,np.matmul(S,R_inv))) # COV = RSSR^-1
    return COV

def f_i(p, D, s, p_i):
    mu = p_i +s*0.5
    COV = Covarianza(D,s)
    constante = (1/2*np.pi)/np.sqrt(np.linalg.det(COV))
    exponente = np.matmul(np.transpose(p-mu), np.matmul(np.linalg.inv(COV), p-mu))
    return constante*np.exp(-exponente/2)

def I_i(x,y, D, s, p_i):
    p = np.array([x,y])
    return f_i(p, D, s, p_i)/f_i(p_i, D, s,p_i)

def logistic(t):
    return 1/(1+np.exp(-t)) 

def PC(x,y, atacantes, defensas, Distancias, Velocidades, Posiciones):
    Influencia_Equipo_1 = 0
    for i in range(0,atacantes): 
        Influencia_Equipo_1 += I_i(x,y, Distancias[i], Velocidades[i], Posiciones[i])

    Influencia_Equipo_2 = 0
    for i in range(atacantes,atacantes + defensas): 
        Influencia_Equipo_2 += I_i(x,y, Distancias[i], Velocidades[i], Posiciones[i])

    return logistic(Influencia_Equipo_1- Influencia_Equipo_2)

#################### Modelo de Probabilidad de Gol #####################

def probabilidad_de_gol(x,y):
        h = 4400+100  # largo de la cancha + margen
        w = 6800 +100+100 # ancho de la cancha + márgenes
        b = 732 # largo de la portería
        numerador = b*(h-y)
        denominador = (((w-b-2*x)**2/4 + (h-y)**2)*((w+b-2*x)**2/4 + (h-y)**2))**0.5
        angulo = np.arcsin(numerador/denominador)
        probabilidad = 1/(1+1/3*np.exp((-angulo+2*np.arctan(3.66/11))/0.1)) 
        return probabilidad   

#################### Simulacion de Contragolpe #######################
id_posesion = 2 
compañero_a_pasar = -1

def normalizacion(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm



def seguridad_de_pase(x,y, a, b, c, d, e, metros): # este debe ser un número entre 0 y 1 
    xlist = np.linspace(x-100*metros/2, x+100*metros/2, 16)
    ylist = np.linspace(y, y + 100*metros, 16)
    X, Y = np.meshgrid(xlist, ylist)
    vPC = np.vectorize(PC, excluded= ['atacantes', 'defensas', 'Distancias', 'Velocidades', 'Posiciones'])
    Z = vPC(X,Y, atacantes = a, defensas = b, Distancias = c , Velocidades = d , Posiciones = e)
    positivos = 0
    for i in range(0,16):
        for j in range(0,16):
            if Z[i][j]>=0.5:    # revisar esto 
                positivos += 1
    return positivos/256

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
        global id_posesion
        global compañero_a_pasar
        id = self.id #el id comienza a contar desde 2
        pos = self.espacio.positions.values()
        pos = np.array(list(pos)).T
        Balon_x = self.model.agents.log[id_posesion-2]['posiciones'][-1][0]
        Balon_y = self.model.agents.log[id_posesion-2]['posiciones'][-1][1]
        pos_Balon = np.array([Balon_x, Balon_y])
        Distancias = [np.linalg.norm(np.array([pos[0][i], pos[1][i]]) - pos_Balon) for i in range(0, self.p.atacantes + self.p.defensas)] #distancias a la pelota. NOTA estamos usando posiciones dinámicas, no las del frame anterior.
        Velocidades = [self.model.agents.log[i]['velocidad_nueva'][-1] for i in range(0, self.p.atacantes + self.p.defensas)]
        Posiciones  = [self.model.agents.log[i]['posiciones'][-1] for i in range(0, self.p.atacantes + self.p.defensas)]
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
        elif self.estado == 0 and 2<=id<=4:
            if id == compañero_a_pasar + 2:
                estado_nuevo = 1
                id_posesion = id
            else: 
                estado_nuevo = 0      

    ###### Atacante con Balón #######
        elif self.estado == 1: 
                       
            #Primero checa la probabilidad de gol desde su posicion y desde la de sus compañeros
            P_gol = np.zeros(self.p.atacantes)
            P_gol_propia = probabilidad_de_gol(self.pos[0], self.pos[1])
            for atacante in range(self.p.atacantes):
                P_gol[atacante] = probabilidad_de_gol(pos[0][atacante], pos[1][atacante])

            # Luego, evalúa si está suficientemente confiado para disparar
            confianza = 0.7
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
                    P_pase_compañero = seguridad_de_pase(pos[0][compañero], pos[1][compañero], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6) #tomamos un área cuadrada de 6*6 metros
                if P_gol_propia*individualidad - P_pase_compañero*(1-individualidad)< 0: #este sería un número entre 1 y -1
                        if compañero_a_pasar == -1: 
                            compañero_a_pasar = compañero
                        else:
                            P_pase_compañero_a_pasar = seguridad_de_pase(pos[0][compañero_a_pasar], pos[1][compañero_a_pasar], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6)
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + P_pase_compañero_a_pasar*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + P_pase_compañero*0.5
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
                    P_pase_compañero = seguridad_de_pase(pos[0][compañero], pos[1][compañero], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones,6)
                    if P_pase_compañero> individualidad: 
                        if compañero_a_pasar == -1:
                            compañero_a_pasar = compañero
                        else:
                            P_pase_compañero_a_pasar = seguridad_de_pase(pos[0][compañero_a_pasar], pos[1][compañero_a_pasar], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6)
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + P_pase_compañero_a_pasar*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + P_pase_compañero*0.5
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
                    P_pase_compañero = seguridad_de_pase(pos[0][compañero], pos[1][compañero], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones,6)
                    if pos[1][compañero]>self.pos[1] and P_pase_compañero>individualidad:
                        if compañero_a_pasar == -1:
                            compañero_a_pasar = compañero
                        else:
                            P_pase_compañero_a_pasar = seguridad_de_pase(pos[0][compañero_a_pasar], pos[1][compañero_a_pasar], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones,6)
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + P_pase_compañero_a_pasar*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + P_pase_compañero*0.5
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
        
        elif self.estado == 2:
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
        pos_Balon = np.array([Balon_x, Balon_y])
        Distancias = [np.linalg.norm(np.array([pos[0][i], pos[1][i]]) - pos_Balon) for i in range(0, self.p.atacantes + self.p.defensas)] #distancias a la pelota. NOTA estamos usando posiciones dinámicas, no las del frame anterior.
        Posiciones  = [self.model.agents.log[i]['posiciones'][-1] for i in range(0, self.p.atacantes + self.p.defensas)]
        if self.model.t <= 1: 
            Velocidades = [np.array([0,60]) for i in range(0, self.p.atacantes + self.p.defensas)]
        elif self.model.t> 1: 
            Velocidades = ['' for i in range(0, self.p.jugadores)]
            for i in range(0, self.p.jugadores):
                x = self.model.agents.log[i]['velocidad_nueva'][-1]
                if  x is None:
                    Velocidades[i] = self.model.agents.log[i]['velocidad_nueva'][-2]
                else: 
                    Velocidades[i] = x
            #print("t = " + str(self.model.t), Velocidades)
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
            
        elif self.estado == 2: 
            velocidad_nueva = np.array([0,0])
           
    ############## Atacante sin Balon ############
        elif self.estado == 0 and 2<= id<= 4:
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

            velocidad_nueva = 60*normalizacion(v_1+v_2+v_3) # aquí el 60 es un comodin para la velocidad maxima


    ############## Defensa ################

        elif 5 <= id : 
            Rapidez_max = [6.8,6.2] # Estas son las velocidades máximas en m/s
            if self.estado == 0: 
                """""    
                El defensa tomará en cuenta a dos contrincantes para decidir cómo se moverá:
                (1) el jugador que tiene el balón y (2) el atacante sin balón más cercano al defensa.
                Para definir las marcas se tomarán las posibles asignaciones 1 a 1 entre el 
                conjunto de defensas y un subconjunto de atacantes sin balón. De entre todas estas
                seleccionaremos la asignación óptima.
                """""    
                
                Asignaciones = funciones_biyectivas(self.p.jugadores, self.p.defensas, id_posesion)
                asignacion_optima = []
                suma_asignacion_optima = 0
                for asignacion in Asignaciones:
                    suma_asignacion = 0
                    for pareja in asignacion:
                        suma_asignacion +=  distancia_entre_agentes(pareja[0], pareja[1], pos)
                    if suma_asignacion > suma_asignacion_optima:
                        asignacion_optima = asignacion
                for pareja in asignacion_optima:
                    if self.id in pareja:
                        id_marca = pareja[1]
                        

                """""
                Una vez asignada su marca, lo primero que hará el defensa será evaluar el valor de 
                la posición de su marca y de quién tenga el balón.
                El valor de la posición será un tipo de promedio entre su seguridad de pase y su 
                probablidad de gol.
                """""

                valor_de_posicion = np.zeros(self.p.atacantes + 2) # se pone el "+2" solo para ser consistente con los índices de los agentes
                seguridad_atacante_balon= seguridad_de_pase(pos[0][id_posesion-2], pos[1][id_posesion-2], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 2)
                valor_de_posicion[id_posesion] = seguridad_atacante_balon*0.4 + probabilidad_de_gol(pos[0][id_posesion -2], pos[1][id_posesion -2])*0.6
                seguridad = seguridad_de_pase(pos[0][id_marca-2], pos[1][id_marca-2], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6)
                valor_de_posicion[id_marca] = seguridad*0.5 + probabilidad_de_gol(pos[0][id_marca-2], pos[1][id_marca-2])*0.5
        
                """""
                A continuación, definirá para su marca y el jugador que tiene el balón, de manera individual
                cuál es la dirección en la que debe moverse para minimizar la seguridad del atacante.
                """""
                theta, direccion_0 = angulo_entre_agentes(self.id, id_marca, pos)
                Rotacion = Matriz_Rotacion(np.array([np.cos(theta/4), np.sin(theta/4)]))
                direcciones = []
                seguridad_marca_minima = 1
                v_1 = np.array([0,0])
                for k in range(0,2):
                    direcciones.append(np.matmul(np.linalg.matrix_power(Rotacion, k), direccion_0))
                    Velocidades_modificado = Velocidades
                    Velocidades_modificado[self.id-2] = 60*direcciones[k] #aquí 60 representa la magnitud de la velocidad
                    seguridad_k = seguridad_de_pase(pos[0][id_marca-2], pos[1][id_marca-2], self.p.atacantes, self.p.defensas, Distancias, Velocidades_modificado, Posiciones, 6)
                    if seguridad_k < seguridad_marca_minima:
                        seguridad_marca_minima = seguridad_k
                        v_1 = 60*direcciones[k]

                theta, direccion_0 = angulo_entre_agentes(self.id, id_posesion, pos)
                Rotacion = Matriz_Rotacion(np.array([np.cos(theta/4), np.sin(theta/4)]))
                direcciones = []
                seguridad_posesion_minima = 1
                v_2 = np.array([0,0])
                for k in range(0,2):
                    direcciones.append(np.matmul(np.linalg.matrix_power(Rotacion, k), direccion_0))
                    Velocidades_modificado = Velocidades
                    Velocidades_modificado[self.id-2] = 60*direcciones[k] #aquí 60 representa la magnitud de la velocidad
                    seguridad_k = seguridad_de_pase(pos[0][id_posesion-2], pos[1][id_posesion-2], self.p.atacantes, self.p.defensas, Distancias, Velocidades_modificado, Posiciones, 2)
                    if seguridad_k < seguridad_posesion_minima:
                        seguridad_posesion_minima = seguridad_k
                        v_2 = 60*direcciones[k]

                v_3 = np.array([0,10])
                
            velocidad_nueva = 60*normalizacion(v_1+v_2+v_3)
  
        self.espacio.move_by(self, velocidad_nueva)
        self.record('velocidad_nueva', velocidad_nueva)
        #print(velocidad_nueva, self.id)
        
        
         
    # def actualizacion(self): 
    #     estado_nuevo = self.model.agents.log[self.id-2]['estado'][-1]
    #     velocidad_nueva = self.model.agents.log[self.id-2]['velocidad_nueva'][-1]
    #     self.estado = estado_nuevo
    #     self.espacio.move_by(self, velocidad_nueva)

            

class Jugada_modelo(ap.Model):

    def setup(self):
        self.espacio = ap.Space(self, shape=[self.p.size]*self.p.dimension)
        self.agents = ap.AgentList(self, self.p.jugadores, Jugador_Simulado) #creamos una cantidad |jugadores| de agentes.
        self.espacio.add_agents(self.agents, [[3450, 650],[2000, 1000], [4200, 1000], [2300, 2000], [4600, 2000]]) #metemos a los agentes creados en el espacio.
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
    animation = ap.animate(m(p), fig, ax,  animacion_individual, **{"interval": 100} ) # interval: 100 quiere decir que entre dos frames hay 100ms.
    plt.show()                                                                         # En otras palabras, la simulación corre a 10 FPS.
    animation.save("Simulacion movimiento 2.gif", "GIF")
    return animation

parameters = {
    'dimension': 2,
    'size': 7000,
    'seed': 123,
    'steps': 60,
    'jugadores': 5,
    'atacantes': 3, 
    'defensas': 2,
    'individualidades': np.random.uniform(0,0.5,3),
    'desbordar': np.random.binomial(1,0.5),
}

animacion_completa(Jugada_modelo, parameters)


modelo = Jugada_modelo(parameters)
resultado = modelo.run()
