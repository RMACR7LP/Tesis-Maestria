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
    return 1/(1+np.exp(-t)) 


def PC(x,y, atacantes, defensas, Distancias, Velocidades, Posiciones):
    Influencia_Equipo_1 = 0
    for i in range(0,atacantes): 
        Influencia_Equipo_1 += I_i(x,y, Distancias[i], Velocidades[i], Posiciones[i])

    Influencia_Equipo_2 = 0
    for i in range(atacantes,atacantes+defensas): 
        Influencia_Equipo_2 += I_i(x,y, Distancias[i], Velocidades[i], Posiciones[i])

    return Influencia_Equipo_1
    #return logistic(Influencia_Equipo_1- Influencia_Equipo_2)

def Mapa_de_Control():
    xlist = np.linspace(-12, 12, 50)
    ylist = np.linspace(-12, 12, 50)
    X, Y = np.meshgrid(xlist, ylist)
    vPC = np.vectorize(PC, excluded= ['atacantes', 'defensas', 'Distancias', 'Velocidades', 'Posiciones'])
    Z = vPC(X,Y,atacantes = 1, defensas = 0, Distancias = np.array([0]), Velocidades = np.array([[0,0]]), Posiciones = np.array([[0,0]]))#, Distancias = np.array([0, np.sqrt(5), 2]), Velocidades = np.array([[0,6],[0,6], [0,-3]]), Posiciones = np.array([[0,0],[-1,2], [0,2]]))
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z, np.linspace(0,1, 15), vmin = 0, vmax = 1)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.scatter(0,0, s=60, c='black', marker = "o")
    ax.scatter(0,0, s=10, c='white', marker = "o")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.show()



Mapa_de_Control()