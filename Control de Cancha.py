import numpy as np
import matplotlib.pyplot as plt

#Distancias = [10,10]# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#Velocidades = np.array([[0,6],[0,6]])#np.array([[0,0], [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0], [0,0],[0,0]])
#Posiciones = np.array([[0,0],[-3,0]])# np.array([[0,0], [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0], [0,0],[0,0]])

# D = 15
# s = np.array([6.36*np.sqrt(2)/2,6.36*np.sqrt(2)/2])
# p_i = np.array([0,0])


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
    mu_i = p_i + s*0.5
    p = np.array([x,y])
    return f_i(p, D, s, p_i)/f_i(mu_i, D, s, p_i)

def logistic(t):
    return 1/(1+np.exp(-3*t)) 

def PC(x,y, atacantes, defensas, Distancias, Velocidades, Posiciones):
    Influencia_Equipo_1 = 0
    for i in range(0,atacantes): 
        Influencia_Equipo_1 += I_i(x,y, Distancias[i], Velocidades[i], Posiciones[i])

    Influencia_Equipo_2 = 0
    for i in range(atacantes,atacantes+defensas): 
        Influencia_Equipo_2 += I_i(x,y, Distancias[i], Velocidades[i], Posiciones[i])

    return logistic(Influencia_Equipo_1 - Influencia_Equipo_2)

def Mapa_de_Control():
    xlist = np.linspace(1900, 5300, 50) #cambiar aqui
    ylist = np.linspace(550, 2100, 50)
    X, Y = np.meshgrid(xlist, ylist)
    vPC = np.vectorize(PC, excluded= ['atacantes', 'defensas', 'Distancias', 'Velocidades', 'Posiciones'])
    pos_Balon = np.array([3450, 650])
    Posiciones = [[3450, 650],[2000, 1000], [5200, 1000], [2300, 2000], [4600, 2000]]
    Distancias = np.array([np.linalg.norm(Posiciones[i] - pos_Balon) for i in range(0, 5)]) #distancias a la pelota. NOTA estamos usando posiciones dinámicas, no las del frame anterior.
    Velocidades = [np.array([19.71484272, 56.66855368]), np.array([32.72625602, 50.28908596]), np.array([27.32225383, 53.41810971]), np.array([  6.7015486 , -59.62456915]), np.array([-11.90960757, -58.80613274])]
    Z = vPC(X,Y,atacantes = 3, defensas = 2, Distancias = Distancias, Velocidades = Velocidades, Posiciones = Posiciones)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    ax.set_xlim(1900, 5300)
    ax.set_ylim(550, 2100)
    cp = ax.contourf(X, Y, Z, np.linspace(0, 1, 15), vmin = 0, vmax = 1)
    fig.colorbar(cp) # Add a colorbar to a plot
    for atacante in range(3):
        ax.scatter(Posiciones[atacante][0], Posiciones[atacante][1], s=60, c = 'black', marker = "o")
    for defensa in range(3,5):
        ax.scatter(Posiciones[defensa][0], Posiciones[defensa][1], s=60, c='blue', marker = "o")
    ax.scatter(pos_Balon[0], pos_Balon[1],  s=10, c='white', marker = "o")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    print(np.median(Z))
    plt.show()

def Mapa_de_Control_2():
    xlist = np.linspace(30.5, 38.5 , 16) #cambiar aqui
    ylist = np.linspace(6.5, 10.5, 16)
    X, Y = np.meshgrid(xlist, ylist)
    vPC = np.vectorize(PC, excluded= ['atacantes', 'defensas', 'Distancias', 'Velocidades', 'Posiciones'])
    pos_Balon = np.array([34.50, 6.50])
    Posiciones = [[34.50, 6.50],[20, 10], [52.00, 10.00], [23.00, 20.00], [46.00, 20.00]]
    Distancias = np.array([np.linalg.norm(Posiciones[i] - pos_Balon) for i in range(0, 5)]) #distancias a la pelota. NOTA estamos usando posiciones dinámicas, no las del frame anterior.
    Velocidades = [np.array([1.971484272, 5.666855368]), np.array([3.272625602, 5.028908596]), np.array([2.732225383, 5.341810971]), np.array([  0.67015486 , -5.962456915]), np.array([-1.190960757, -5.880613274])]
    Z = vPC(X,Y,atacantes = 3, defensas = 2, Distancias = Distancias, Velocidades = Velocidades, Posiciones = Posiciones)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    ax.set_xlim(15.00, 58.00)
    ax.set_ylim(5.50, 21.00)
    cp = ax.contourf(X, Y, Z, np.linspace(0, 1, 15), vmin = 0, vmax = 1)
    fig.colorbar(cp) # Add a colorbar to a plot
    for atacante in range(3):
        ax.scatter(Posiciones[atacante][0], Posiciones[atacante][1], s=60, c = 'black', marker = "o")
    for defensa in range(3,5):
        ax.scatter(Posiciones[defensa][0], Posiciones[defensa][1], s=60, c='blue', marker = "o")
    ax.scatter(pos_Balon[0], pos_Balon[1],  s=10, c='white', marker = "o")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    print(np.median(Z))
    plt.show()

Mapa_de_Control_2()