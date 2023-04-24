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
ficadores es [2,3,4,5,6] en una simulación donde hay 5 agentes. Por ello en
la función setup de la clase jugador simulado está presente la linea
self.id = self.id-2.


2) La simulación corre a 10 FPS, es decir 1 segundo de la vida real equivale
a 10 fotogramas de la simulación.


"""""


#################### Funciones Auxiliares ############################

"""""
La siguiente función permite hallar todas las funciones biyectivas
el conjunto de defensas y un subconjunto de los atacantes que no contenga
al jugador con posesión del balón. Si los jugadores son {0,1,2,3,4} y están
divididos en atacantes = {0,1,2} y defensas = {3,4}, y el jugador con 
posesión tiene id = 2, entonces el output de esta función es

                    [[(3, 0), (4, 1)], [(3, 1), (4, 0)]]

"""""

def funciones_biyectivas(numero_jugadores, numero_defensas, posesión):
      atacantes = numero_jugadores- numero_defensas
      iterable = ''
      for atacante in range(0, atacantes):
            if atacante != posesión: 
                  iterable += str(atacante)
      
      permutaciones = itertools.permutations(iterable, numero_defensas)
      x = np.arange(atacantes, numero_jugadores)
      funciones = []
      for perm in permutaciones:
            funcion = []
            for i in range(np.size(x)):
                  funcion.append((x[i], int(perm[i])))
            funciones.append(funcion)
      
      return funciones

def distancia_entre_agentes(id_agente_1, id_agente_2, matriz_de_posiciones): 
    """""
    La matriz de posiciones será la de agentpy, que corre de 0 al número de jugadores
    tanto en sus filas como en las columnas. Además la división dentro de 100 al final 
    hace que esta función retorne la distancia en metros de la "vida real".
    """""
    distancia = np.linalg.norm(matriz_de_posiciones[id_agente_1] - matriz_de_posiciones[id_agente_2])
    return distancia/100  

def angulo_entre_agentes(id_agente_1, id_agente_2, matriz_de_posiciones):
    pos_defensa = matriz_de_posiciones[id_agente_1]
    pos_jugador_con_balon = matriz_de_posiciones[id_agente_2]
    vector_1 = pos_jugador_con_balon - pos_defensa
    vector_1 = vector_1/np.linalg.norm(vector_1)
    vector_2 = np.array([0,1])
    angulo =  np.arccos(np.dot(vector_1, vector_2))
    if vector_1[0] < 0: 
        angulo = - angulo
    return angulo, vector_1

def duracion_pase(d):

    """""
    Dependiendo de la distancia entre el emisor y el receptor de los pases variará el tiempo que 
    tarda en recorrer la pelota entre ambos puntos. El tiempo que se tardará la pelota seguirá la 
    siguiente regla:

    || t  (segundos) ||   0.5    ||    1    ||   1.5    ||    2    ||
    ----------------------------------------------------------------
    || d (metros)    ||  [0, 8)  ||  [8,16) || [16, 25) || [25,35) ||  

    La función retornará la cantidad de frames que tomará en realizarse el pase.

    """""
    if 0 <= d < 8: 
        return  5
    elif 8 <= d < 20:
        return 10
    elif 20 <= d < 30:
        return 15
    elif 30<= d < 40: 
        return 20
    else: 
        return 25 

def trayectoria_de_gol(pos_pateador, pos_defensa):
    x_1, y_1= pos_pateador
    x_2, y_2 = pos_defensa
    pendiente_trayectoria = (y_2-y_1)/(x_2-x_1)
    x = (4500-y_1)/pendiente_trayectoria + x_1
    if y_2 > y_1 and 3134 < x < 3866:
        return True


#################### Modelo de Control de Cancha #####################

"""
Nota: En el modelo de control de cancha las escalas están 1:1
es decir, R = 10 representa una distancia de 10 metros. Por otro lado,
en el espacio de la simulacion, la relacion es 100:1. Es decir R = 100
representa una distancia de 1 metro.
"""

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
    mu_i = p_i +s*0.5
    COV = Covarianza(D,s)
    constante = (1/2*np.pi)/np.sqrt(np.linalg.det(COV))
    exponente = np.matmul(np.transpose(p-mu_i), np.matmul(np.linalg.inv(COV), p-mu_i))
    return constante*np.exp(-exponente/2)

def I_i(x,y, D, s, p_i):
    mu_i = p_i+s*0.5
    p = np.array([x,y])
    return f_i(p, D, s, p_i)/f_i(mu_i, D, s,p_i)

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

def probabilidad_de_gol_libre(pos_jugador):
    alpha = 1.7
    """""
    Primero hacemos una traslacion para "colocar" el centro de la portería en el punto (0,0) escalar
    apropiadamente (dividiendo dentro de 100) para así poder utilizar las mismas ecuaciones que en la 
    tesina. La transformación es entonces (x,y) -> (x-3400, y-4500)/100
    """""  
    x = pos_jugador[0]
    y = pos_jugador[1]    
    x = (x-3500)/100
    y = (y-4500)/100
    r_0 = 1/alpha * np.sqrt( (x**2 + (alpha*y)**2 -3.66**2)**2/(2*alpha*y)**2 + 3.66**2 )
    d = r_0 + r_0*np.sqrt(1-3.66**2/(alpha*r_0)**2)

    probabilidad = 1/(1+1/3*np.exp((-2*np.arctan(3.66/d)+2*np.arctan(3.66/11))/0.1)) 
    return probabilidad   

def probabilidad_de_gol(pos_jugador, thetas_cubiertos):
    A = pos_jugador
    B = np.array([3134, 4500])
    C = np.array([3866, 4500])
    AB = np.linalg.norm(A-B)
    AC = np.linalg.norm(A-C) 
    theta_portería = np.arccos(-(40000 - AB**2 - AC**2)/(2*AB*AC))
    probabilidad = probabilidad_de_gol_libre(pos_jugador)
    p_intercepción = 0.8 # valor aleatorio
    for theta_cubierto in thetas_cubiertos:
        probabilidad *= p_intercepción*(theta_cubierto/theta_portería)
    return probabilidad

#################### Simulacion de Contragolpe #######################
id_posesion = 1 
compañero_a_pasar = -1
tiempo_pase = 0

def normalizacion(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def seguridad_de_pase(punto, a, b, c, d, e, metros): # este debe ser un número entre 0 y 1 
    x = punto[0]
    y = punto[1]
    xlist = np.linspace(x-100*metros/2, x+100*metros/2, 16)
    ylist = np.linspace(y, y + 100*metros, 16)
    X, Y = np.meshgrid(xlist, ylist)
    vPC = np.vectorize(PC, excluded= ['atacantes', 'defensas', 'Distancias', 'Velocidades', 'Posiciones'])
    Z = vPC(X,Y, atacantes = a, defensas = b, Distancias = c , Velocidades = d , Posiciones = e)
    return np.median(Z)

class Jugador_Simulado(ap.Agent):  

    def setup(self):
        self.id = self.id-2
        id = self.id
        global id_posesion
        if id == id_posesion: 
            self.estado = 1
        else: 
            self.estado = 0
        self.record('estado', self.estado)
    
    def setup_pos_s(self, espacio):
        self.espacio = espacio
        self.pos = espacio.positions[self]   # .positions es una variable de la clase Space, es un diccionario que vincula a cada agente con sus coordenadas en el espacio.
        self.record('posiciones', np.array([self.pos[0], self.pos[1]]))

    def cambio_estado(self): 
        global id_posesion
        global compañero_a_pasar
        global tiempo_pase
        id = self.id 
        # pos = self.espacio.positions.values()
        # pos = np.array(list(pos)).T
        Balon_x = self.model.agents.log[id_posesion]['posiciones'][-1][0]
        Balon_y = self.model.agents.log[id_posesion]['posiciones'][-1][1]
        pos_Balon = np.array([Balon_x, Balon_y])
        Posiciones = self.espacio.positions.values()
        Posiciones = np.array(list(Posiciones)).T
        Posiciones  = np.array([self.model.agents.log[i]['posiciones'][-1] for i in range(0, self.p.atacantes + self.p.defensas)])
        Distancias = np.array([np.linalg.norm(Posiciones[i] - pos_Balon) for i in range(0, self.p.atacantes + self.p.defensas)]) #distancias a la pelota. NOTA estamos usando posiciones dinámicas, no las del frame anterior.
        Velocidades = np.array([self.model.agents.log[i]['velocidad_nueva'][-1] for i in range(0, self.p.atacantes + self.p.defensas)])
        
        
        # Definiremos tener posesion de la pelota como estado = 1, no tenerla como estado = 0
        # y disparar como estado 2.

    
    ########### Defensa #############    
        if self.estado == 0 and self.p.atacantes <= id:
            estado_nuevo = 0

    ###### Atacante Sin Balón #######
        elif self.estado == 0 and 0 <= id < self.p.atacantes:
            if id == compañero_a_pasar:
                id_posesion = id
                if tiempo_pase>0: 
                    tiempo_pase -= 1
                    estado_nuevo = 0
                elif tiempo_pase == 0: 
                    for jugador in range(self.p.atacantes,self.p.jugadores):
                        if distancia_entre_agentes(self.id, jugador, Posiciones) <= 3:
                            seguridad_posesion = seguridad_de_pase(self.pos, self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 1.5)
                            volado = np.random.binomial(1, 0.01) #seguridad_posesion)
                            if volado == 0: 
                                estado_nuevo = -1
                            else:
                                estado_nuevo = 1
                        else: 
                            estado_nuevo = 1
            else: 
                estado_nuevo = 0      

    ###### Atacante con Balón #######
        elif self.estado == 1: 
                       
            #Primero checa la probabilidad de gol desde su posicion y desde la de sus compañeros
            P_gol = np.zeros(self.p.atacantes)
            P_gol_propia = probabilidad_de_gol_libre(self.pos)
            for atacante in range(self.p.atacantes):
                P_gol[atacante] = probabilidad_de_gol_libre(Posiciones[atacante])

            # Luego, evalúa si está suficientemente confiado para disparar
            confianza = 0.7
            if P_gol_propia > confianza: 
                confiado = 1
            else: 
                confiado = 0
            
            #A continuación evalúa si hay un compañero con mayor probabilidad de gol
            Mejor_compañero = []
            for atacante in range(self.p.atacantes):
                if P_gol_propia < P_gol[atacante]:
                    Mejor_compañero.append(atacante)

            # Ahora, hay 4 escenarios posibles que dependen de (a) si está confiado para disparar
            # y de (b) si hay un compañero con mayor probabilidad de gol: 
            
            individualidad = self.p.individualidades[id]
            # Caso 1: a sí y b sí.  
            if confiado == 1 and len(Mejor_compañero)>0:
                conducir = 0
                compañero_a_pasar = -1
                for compañero in Mejor_compañero:
                    P_pase_compañero = seguridad_de_pase(Posiciones[compañero], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6) #tomamos un área cuadrada de 6*6 metros
                if P_gol_propia*individualidad - P_pase_compañero*(1-individualidad)< 0: #este sería un número entre 1 y -1
                        if compañero_a_pasar == -1: 
                            compañero_a_pasar = compañero
                        else:
                            P_pase_compañero_a_pasar = seguridad_de_pase(Posiciones[compañero_a_pasar], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6)
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + P_pase_compañero_a_pasar*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + P_pase_compañero*0.5
                            if efectividad_2>efectividad_1:
                                compañero_a_pasar = compañero
                if compañero_a_pasar == -1: 
                    disparar = 1
                elif compañero_a_pasar != -1:
                    disparar = 0
                    tiempo_pase = duracion_pase(distancia_entre_agentes(compañero_a_pasar, self.id, Posiciones))


            # Caso 2: a sí y b no.
            elif confiado == 1 and len(Mejor_compañero) == 0:
                disparar = 1
                conducir = 0
                
            # Caso 3: a no y b sí:
            elif confiado == 0 and len(Mejor_compañero)>0:
                disparar = 0
                compañero_a_pasar = -1
                for compañero in Mejor_compañero:
                    P_pase_compañero = seguridad_de_pase(Posiciones[compañero], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones,6)
                    if P_pase_compañero> individualidad: 
                        if compañero_a_pasar == -1:
                            compañero_a_pasar = compañero
                        else:
                            P_pase_compañero_a_pasar = seguridad_de_pase(Posiciones[compañero_a_pasar], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6)
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + P_pase_compañero_a_pasar*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + P_pase_compañero*0.5
                            if efectividad_2>=efectividad_1:
                                compañero_a_pasar = compañero
                if compañero_a_pasar == -1:
                    conducir = 1
                elif compañero_a_pasar != -1:
                    conducir = 0
                    tiempo_pase = duracion_pase(distancia_entre_agentes(compañero_a_pasar, self.id, Posiciones))

            # Caso 4: a no y b no: 
            else: 
                disparar = 0
                compañero_a_pasar = -1
                for compañero in range(self.p.atacantes):
                    P_pase_compañero = seguridad_de_pase(Posiciones[compañero], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones,6)
                    if Posiciones[compañero][1] > self.pos[1] and P_pase_compañero>individualidad:
                        if compañero_a_pasar == -1:
                            compañero_a_pasar = compañero
                        else:
                            P_pase_compañero_a_pasar = seguridad_de_pase(Posiciones[compañero_a_pasar], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones,6)
                            efectividad_1 = P_gol[compañero_a_pasar]*0.5 + P_pase_compañero_a_pasar*0.5 #podriamos poner un parametro en ves del 0.5
                            efectividad_2 = P_gol[compañero]*0.5 + P_pase_compañero*0.5
                            if efectividad_2>efectividad_1:
                                compañero_a_pasar = compañero
                if compañero_a_pasar == -1: 
                    conducir = 1
                elif compañero_a_pasar != -1:
                    conducir = 0
                    tiempo_pase = duracion_pase(distancia_entre_agentes(compañero_a_pasar, self.id, Posiciones))

            # Y aquí estan las reglas de cambio de estado basado en los cuatro casos mencionados
            
            if conducir == 1:
                estado_nuevo = 1
            
            if disparar == 1: 
                estado_nuevo = 2

            if conducir == 0 and disparar == 0: 
                estado_nuevo = 0

            # También tomamos en cuenta la posibilidad de que un defensa suficientemente cerca 
            # pueda quitarle la pelota al jugador con posesión.
            if self.model.t % 5 == 0:
                for defensa in range(self.p.atacantes,self.p.jugadores):
                    if distancia_entre_agentes(self.id, defensa, Posiciones) <= 1:
                        seguridad_posesion = seguridad_de_pase(self.pos, self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 1.5)
                        volado = np.random.binomial(1, seguridad_posesion)
                        if volado == 0: 
                            estado_nuevo = -1

        elif self.estado == 2:
            estado_nuevo = 2
        
        elif self.estado == -1:
            estado_nuevo = -1

        
        self.record('estado', estado_nuevo)
        self.estado = estado_nuevo
        

    def cambio_velocidad(self):
        id = self.id #el id comienza a contar desde 2
        # pos = self.espacio.positions.values()
        # pos = np.array(list(pos)).T
        #self.record('posiciones', np.array([self.pos[0], self.pos[1]]))
        Balon_x = self.model.agents.log[id_posesion]['posiciones'][-1][0]
        Balon_y = self.model.agents.log[id_posesion]['posiciones'][-1][1]
        pos_Balon = np.array([Balon_x, Balon_y])
        Posiciones  = [self.model.agents.log[i]['posiciones'][-1] for i in range(0, self.p.atacantes + self.p.defensas)]
        Distancias = [np.linalg.norm(np.array(Posiciones[i] - pos_Balon)) for i in range(0, self.p.atacantes + self.p.defensas)] #distancias a la pelota. NOTA estamos usando posiciones dinámicas, no las del frame anterior.
        
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
   
    ############## Atacante con Balon ###########
        if self.estado == 1:
            individualidad = self.p.individualidades[id]
            # Primero haremos como que el que tiene la pelota siempre va a la direccion del punto medio de la porteria            
            v_1 = normalizacion(np.array([3500-self.pos[0],4500- self.pos[1]]))*individualidad
            v_2 = np.array([0,1])*(1-individualidad)
            velocidad_nueva = 60*normalizacion(v_1+v_2) # el 60 está como comodin para la magnitud de la velocidad
            
        elif self.estado == 2: 
            velocidad_nueva = np.array([0,0])
        
        elif self.estado == -1:
            velocidad_nueva = np.array([0,0])
           
    ############## Atacante sin Balon ############
        elif self.estado == 0 and 0 <= id < self.p.atacantes:
            individualidad = self.p.individualidades[id] # esto controla si el jugador busca recibir pase para centrar o un pase para disparar.
            v_1 = normalizacion(np.array([3500-self.pos[0],4500- self.pos[1]]))*individualidad # qué tanto quiere ir a la portería?
            if self.pos[0]<=Balon_x:
                v_2 = normalizacion(np.array([1484-self.pos[0], 4500-self.pos[1]]))*(1-individualidad) # qué tanto se va a abrir para buscar recibir el pase y centrar.
            elif self.pos[0]> Balon_x:
                v_2 = normalizacion(np.array([5516-self.pos[0], 4500-self.pos[1]]))*(1-individualidad)
            

            velocidad_nueva = 60*normalizacion(v_1+v_2) # aquí el 60 es un comodin para la velocidad maxima


    ############## Defensa ################

        elif self.p.atacantes <= id : 
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
                suma_asignacion_optima = np.inf 
                for asignacion in Asignaciones:
                    suma_asignacion = 0
                    for pareja in asignacion:
                        suma_asignacion +=  distancia_entre_agentes(pareja[0], pareja[1], Posiciones)
                    if suma_asignacion < suma_asignacion_optima:
                        asignacion_optima = asignacion
                        suma_asignacion_optima = suma_asignacion
                for pareja in asignacion_optima:
                    if self.id in pareja:
                        id_marca = pareja[1]
                        

                """""
                Una vez hallada la asignación óptima, definirá para su marca y el jugador que tiene el balón, de manera individual,
                cuál es la dirección en la que debe moverse para minimizar la seguridad del atacante. Entre las direcciones
                posibles, será directamente hacia la dirección en la que se encuentra el defensa o theta/4 hacia "arriba" 
                de esa direccion.
                """""
                theta, direccion_0 = angulo_entre_agentes(self.id, id_marca, Posiciones)
                Rotacion = Matriz_Rotacion(np.array([np.cos(theta/4), np.sin(theta/4)]))
                direcciones = []
                seguridad_marca_minima = 1
                v_1 = np.array([0,0])
                for k in range(0,2):
                    direcciones.append(np.matmul(np.linalg.matrix_power(Rotacion, k), direccion_0))
                    Velocidades_modificado = Velocidades
                    Velocidades_modificado[self.id] = 60*direcciones[k] #aquí 60 representa la magnitud de la velocidad
                    seguridad_k = seguridad_de_pase(Posiciones[id_marca], self.p.atacantes, self.p.defensas, Distancias, Velocidades_modificado, Posiciones, 6)
                    if seguridad_k <= seguridad_marca_minima:
                        seguridad_marca_minima = seguridad_k
                        v_1 = direcciones[k]

                theta, direccion_0 = angulo_entre_agentes(self.id, id_posesion, Posiciones)
                Rotacion = Matriz_Rotacion(np.array([np.cos(theta/4), np.sin(theta/4)]))
                direcciones = []
                seguridad_posesion_minima = 1
                v_2 = np.array([0,0])
                for k in range(0,2):
                    direcciones.append(np.matmul(np.linalg.matrix_power(Rotacion, k), direccion_0))
                    Velocidades_modificado = Velocidades
                    Velocidades_modificado[self.id] = 60*direcciones[k] #aquí 60 representa la magnitud de la velocidad
                    seguridad_k = seguridad_de_pase(Posiciones[id_posesion], self.p.atacantes, self.p.defensas, Distancias, Velocidades_modificado, Posiciones, 2)
                    if seguridad_k < seguridad_posesion_minima:
                        seguridad_posesion_minima = seguridad_k
                        v_2 = direcciones[k]


                """""
                Finalmente, el defensa evaluará el "valor" de la posición de su marca y de quién 
                tenga el balón y a partir de este dará más prioridad a una dirección sobre otra entre
                las dos calculadas anteriormente (la marca y quien tiene el balón).
                El valor de la posición será un tipo de promedio entre su seguridad de pase y su 
                probablidad de gol.
                """""

                valor_de_posicion = np.zeros(self.p.atacantes) 
                seguridad_atacante_balon= seguridad_de_pase(Posiciones[id_posesion], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 2)
                valor_de_posicion[id_posesion] = seguridad_atacante_balon*0.4 + probabilidad_de_gol_libre(Posiciones[id_posesion])*0.6 # para quien tiene la posesión debe pesar más su probabilidad de gol
                seguridad = seguridad_de_pase(Posiciones[id_marca], self.p.atacantes, self.p.defensas, Distancias, Velocidades, Posiciones, 6)
                valor_de_posicion[id_marca] = seguridad*0.5 + probabilidad_de_gol_libre(Posiciones[id_marca])*0.5 #para la marca la probabilidad de gol y la seguridad de pase son igual de importantes.
        
                

                """""
                Tomando en cuenta también una dirección vertical para que no se pase.
                """""
                v_3 = np.array([0,0.1])
                agresividad = self.p.agresividades[id-5]

                
            velocidad_nueva = 60*normalizacion((1-agresividad)*valor_de_posicion[id_marca]*v_1 + agresividad*valor_de_posicion[id_posesion]*v_2 + v_3)
        
        ########### Intercepción, Disparo, Offside ######### 
        for jugador in range(self.p.atacantes):
            estado = self.model.agents.log[jugador]['estado'][-1]
            if estado is None: 
                estado = self.model.agents.log[jugador]['estado'][-2]

            if estado in [-1,2]:
                velocidad_nueva = np.array([0,0])

        self.espacio.move_by(self, velocidad_nueva)
        self.record('posiciones', np.array([self.pos[0], self.pos[1]]))
        self.record('velocidad_nueva', velocidad_nueva)

    def Gol(self):
        Posiciones  = [self.model.agents.log[i]['posiciones'][-1] for i in range(0, self.p.jugadores)]
        Gol = 0
        for jugador in range(self.p.jugadores):
            estado = self.model.agents.log[jugador]['estado'][-1]
            if estado is None: 
                estado = self.model.agents.log[jugador]['estado'][-2]

            if estado == 2: 
                """
                Primero verificamos qué defensas están en una posible trayectoria a gol.
                """

                defensas_en_trayectoria = []
                for defensa in range(self.p.atacantes, self.p.defensas):
                    if trayectoria_de_gol(Posiciones[jugador], Posiciones[defensa]):
                        defensas_en_trayectoria.append(defensa)
                """
                Luego, calculamos los ángulos cubiertos por estos defensas

                """

                theta_cubiertos = []
                for defensa in defensas_en_trayectoria:
                    x_1, y_1 = Posiciones[jugador]
                    x_2, y_2 = Posiciones[defensa]
                    D_x = (y_2 - y_1)*(3134 - x_1)/(4500 - y_1) + x_1 
                    E_x = (y_2 - y_1)*(3866 - x_1)/(4500 - y_1) + x_1 
                    A = Posiciones[jugador]
                    B = np.array([max(x_2-1, D_x), y_2])
                    C = np.array([min(x_2+1, E_x), y_2])
                    AB = np.linalg.norm(A-B)
                    AC = np.linalg.norm(A-C) 
                    theta_cubierto = np.arccos((4 - AB**2 - AC**2)/(2*AB*AC))
                    theta_cubiertos.append(theta_cubierto)

                p_gol = probabilidad_de_gol(Posiciones[jugador], theta_cubiertos)
                volado = np.random.binomial(1, p_gol)
                if volado == 1: 
                    Gol = 1
        return Gol


class Jugada_modelo(ap.Model):

    def setup(self):
        self.espacio = ap.Space(self, shape=[self.p.size, 9000])
        self.agents = ap.AgentList(self, self.p.jugadores, Jugador_Simulado) #creamos una cantidad |jugadores| de agentes.
        self.espacio.add_agents(self.agents, [[3450, 650],[2000, 1000], [5200, 1000], [2300, 2000], [4600, 2000]]) #metemos a los agentes creados en el espacio.
        self.agents.setup_pos_s(self.espacio)
        self.espacio.record('Gol', 0)
        

    def step(self): 
        if self.model.t > 0:
            self.agents.cambio_velocidad()    
            self.agents.cambio_estado()
            self.espacio.record('Gol', self.agents.Gol()[0])

def animacion_individual(m, ax):
    ax.set_title(f"{m.t}")
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 5100)
    ax.set_axis_off()
    pos = m.espacio.positions.values()
    pos = np.array(list(pos)).T 
    ax.plot([3134, 3134, 3866, 3866], [4500, 4800, 4800, 4500], color = "white") # portería
    ax.plot([2584, 2584, 4416, 4416], [4500, 3950, 3950, 4500], color = "white") # area pequeña
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
        elif estado == -1: 
             ax.scatter(pos[0][atacante],pos[1][atacante], s=50, c='#808080', marker = "o")


    for defensa in range(m.p.atacantes, 5):
        ax.scatter(pos[0][defensa],pos[1][defensa], s=50, c='black', marker = "o")    
 
    if 1 in m.espacio.log['Gol']: 
        ax.text(2400, 1000, "GOL", fontsize = 100, fontfamily = 'cursive')

def animacion_completa(m, p):
    fig = plt.figure(figsize=(10,7))
    fig.patch.set_facecolor("green")
    ax = fig.add_subplot()
    animation = ap.animate(m(p), fig, ax,  animacion_individual, **{"interval": 100} ) # interval: 100 quiere decir que entre dos frames hay 100ms.
    plt.show()                                                                         # En otras palabras, la simulación corre a 10 FPS.
    animation.save("Simulacion Prueba.gif", "GIF")
    return animation

parameters = {
    'dimension': 2,
    'size': 7000,
    'seed': 123,
    'steps': 60,
    'jugadores': 5,
    'atacantes': 3, 
    'defensas': 2,
    'individualidades': np.array([0.8, 0.8, 0.8]),
    'agresividades': np.array([0.63, 0.58]), # [0.56, 0.65]
}

animacion_completa(Jugada_modelo, parameters)


# modelo = Jugada_modelo(parameters)
# resultado = modelo.run()
