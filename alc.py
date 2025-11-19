import numpy as np
import math
import os
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import time

## Laboratorio 1
tol = 1e-15

def error(x, y):
    """
    Recibe dos números x e y, y calcula el error de aproximar x usando y en float64.
    """
    return abs(x - y)

def error_relativo(x, y):
    """
    Recibe dos números x e y, y calcula el error relativo de aproximar x usando y en float64.
    """
    if abs(x) < tol:   # x muy cercano a 0
        return abs(y)
    return abs(x - y) / abs(x)

def matricesIguales(A, B, tol = 1e-8):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, además de distintos valores.
    """
    if A.shape != B.shape:
        return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if error_relativo(A[i, j], B[i, j]) > tol:
                return False
    return True

## Laboratorio 2

def rota(theta):
    """
    Recibe un ángulo theta y retorna una matriz de 2x2
    que rota un vector dado en un ángulo theta.
    """
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])

def escala(s):
    """
    Recibe una tira de números s y retorna una matriz cuadrada de
    n x n, donde n es el tamaño de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i].
    """
    n = len(s)
    res = np.zeros((n,n))
    for i in range (n):
        res[i][i] = s[i]
    return res

def rota_y_escala(theta, s):
    """
    Recibe un ángulo theta y una tira de números s,
    y retorna una matriz de 2x2 que rota el vector en un ángulo theta
    y luego lo escala en un factor s.
    """
    return multi_matricial(rota(theta), escala(s))

def afin(theta, s, b):
    """
    Recibe un ángulo theta, una tira de números s (en R2), y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un ángulo theta,
    luego lo escala en un factor s y por último lo mueve en un valor fijo b.
    """
    A = rota_y_escala(theta, s)
    T = np.eye(3)
    T[:2,:2] = A    # Inserto A en la parte superior izquierda
    T[:2,2] = b     # Inserto b en la última columna
    return T

def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un ángulo theta,
    una tira de números s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformación afín a v.
    """
    vh = np.array([v[0], v[1], 1])      # vector homogéneo
    wh = multi_matricial(afin(theta, s, b), traspuesta(vh))       # aplicar transformación
    return wh[:2]                       # volver a R2

## Laboratorio 3

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    """
    if p == 'inf':
        return np.max(np.abs(x))
    else:
        return (np.sum(np.abs(x)**p))**(1/p)

def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacíos, y un escalar p. Devuelve
    una lista donde cada elemento corresponde a normalizar los
    elementos de X con la norma p.
    """
    Y = X.copy()
    for i in range(len(Y)):
        vector = Y[i]
        n = norma(vector, p)
        if (n == 0):
            Y[i] = np.zeros_like(vector)
        else:
            Y[i] = (vector / n)
    return Y

def multi_matricial(A, B):
    # Si ambos son vectores 1D → producto interno
    if A.ndim == 1 and B.ndim == 1:
        if A.shape[0] != B.shape[0]:
            raise ValueError("Los vectores no tienen la misma longitud")
        return np.sum(A*B)

    # Si A vector columna y B es vector fila → tratar B como fila (1 x n)
    if A.ndim == 2 and A.shape[1] == 1 and B.ndim == 1:
        B = B.reshape(1, -1)
    # Si A matriz y B es vector fila → tratar B como columna (n x 1)
    if A.ndim == 2 and A.shape[1] > 1 and B.ndim == 1:
        B = B.reshape(-1, 1)
    # Si A vector fila → tratar A como fila (1 x n)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    m, n = A.shape
    k, p = B.shape
    # Chequear dimensiones compatibles
    if n != k:
        raise ValueError(f"No se pueden multiplicar: {(m, n)} y {(k, p)}")

    # 3. Inicializar la matriz resultado (m x p)
    C = np.zeros((m, p))

    # 4. Iterar sobre las filas de A (i) y las columnas de B (j) 
    for i in range(m):
        for j in range(p):

            # # Esto es lo que codeamos originalmente. Por una cuestion de performance tuvimos que reemplazarlo por np.sum 
            # # para que corra el tp en un tiempo razonable. Todos los tests de los labos pasan usando estas dos lineas
            # # en lugar de la de np.sum
            # for k in range(n):
            #     C[i, j] += A[i, k] * B[k, j]

            # El elemento C[i, j] es el producto punto de la fila i de A y la columna j de B.
            C[i, j] = np.sum(A[i, :] * B[:, j])

    # Si el resultado es un vector columna o fila → devolver como 1D
    
    if (C.ndim == 2 and C.shape[0] == 1 and C.shape[1] == 1):
        return C[0]
    if C.shape[1] == 1:
        return C.flatten()

    return C

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q, p} y el vector x en el cual se alcanza
    el máximo.
    """
    norma_max = 0; x_max = None
    for _ in range(Np):
        n = A.shape[1]
        x = 2*np.random.rand(n) - 1
        x_normalizado = x / norma(x, p)
        valor =  norma(multi_matricial(A, traspuesta(x_normalizado)), q)
        if valor > norma_max:
            norma_max = valor
            x_max = x_normalizado
    return norma_max, x_max

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    """
    if p == 1:
        sumas_columnas = np.sum(np.abs(A), axis=0) 
        return np.max(sumas_columnas)
    if p == 'inf':
        sumas_filas = np.sum(np.abs(A), axis=1) 
        return np.max(sumas_filas)

def condMC(A, p, Np):
    """
    Devuelve el número de condición de A usando la norma inducida p.
    """
    inversa = np.linalg.inv(A)
    norma = normaMatMC(A,p,p,Np)
    normaInv = normaMatMC(inversa,p,p,Np)
    return norma[0] * normaInv[0]

def condExacta(A, p):
    """
    Devuelve el número de condición de A a partir de la fórmula de
    la ecuación (1) usando la norma p.
    """
    inversa = np.linalg.inv(A)
    return normaExacta(A,p) * normaExacta(inversa,p)
    return  

## Laboratorio 4

def calculaLU(A):
    """
    Calcula la factorizacion LU de la matriz A y retorna las matrices L
    y U, junto con el numero de operaciones realizadas. En caso de
    que la matriz no pueda factorizarse retorna None.
    """
    nops = 0
    m, n = A.shape
    U = A.copy()
    
    if m != n:
        print('Matriz no cuadrada')
        return None, None, 0
        
    for k in range(n - 1):
        if (U[k, k] == 0): return None, None, 0

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            U[i, k] = factor
            nops += 1
            U[i, k + 1:] = U[i, k + 1:] - factor * U[k, k + 1:]
            nops += 2 * (n - k - 1) 
            
    if (U[n-1, n-1] == 0): return None, None, 0
    L = np.eye(n)
    for j in range(n):
        L[j+1:, j] = U[j+1:, j]
        U[j+1:, j] = 0

    return L, U, nops

def res_tri(L, b, inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar
    si es triangular inferior o superior usando el argumento
    inferior (por default asumir que es triangular inferior).
    """
    n = L.shape[1]
    
    if inferior:
        y = np.zeros(n)
        for i in range(0, n):
            terminos_restar = multi_matricial(L[i, :i], y[:i])
            
            y_i = b[i] - terminos_restar
            y[i] = y_i / L[i, i]
        return y
        
    else: # Triangular Superior (Sustitución Hacia Atrás)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            terminos_restar = multi_matricial(L[i, i+1:], x[i+1:])
            
            x_i = b[i] - terminos_restar
            x[i] = x_i / L[i, i]
        return x

def inversa(A):
    """
    Calcula la inversa de A empleando la factorizacion LU
    y las funciones que resuelven sistemas triangulares.
    """
    L, U, nops = calculaLU(A)
    if (L is None): return None
    n = L.shape[0]
    A_inv = np.zeros((n, n))
    
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1
        y = res_tri(L, e_j, inferior=True)   # L y e_j
        x = res_tri(U, y, inferior=False)    # U y y
        A_inv[:, j] = x                      # columna j de la inversa
    
    return A_inv

def traspuesta(A):
    if (A.ndim == 1):
        return A.reshape(-1, 1)

    n, m = A.shape
    At = np.empty((m, n))

    for i in range(n):
        At[:, i] = A[i, :]
    return At

def vector_traspuesto(v):
    n = v.shape[0]
    vt = np.zeros((n, 1))
    for i in range (n):
        vt[i][0] = v[i]
    return vt

def simetrica(A):
    m=A.shape[0]
    n=A.shape[1]
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    for i in range(n):
        for j in range(n):
            if (A[i][j] != A[j][i]):
                return False
    return True

def calculaLDV(A):
    """
    Calcula la factorizacion LDV de la matriz A, de forma tal que A =
    LDV, con L triangular inferior, D diagonal y V triangular
    superior. En caso de que la matriz no pueda factorizarse
    retorna None.
    """
    L, U, nops1 = calculaLU(A)
    if (L is None): return None, None, None, 0
    Ut = traspuesta(U)
    V, D , nops2= calculaLU(Ut)

    return L, D, traspuesta(V) , nops1 + nops2

def esSDP(A, atol=1e-8):
    """
    Checkea si la matriz A es simetrica definida positiva (SDP) usando
    la factorizacion LDV.
    """
    if (not simetrica(A)): return False

    L, D, V, nops = calculaLDV(A)
    if (D is None): return False
    for i in range(A.shape[0]):
        if (D[i][i] <= 0): return False
    return True

def producto_interno(v, w):
    res = 0
    for i in range(w.shape[0]):
        res += v[i]*w[i]
    return res

## Laboratorio 5

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de nxn
    tol la tolerancia con la que s efiltran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retrona matrices Q y R calculadas con Gram Schmidt (y como tercer argumento 
    opcional, el numero de operaciones)
    Si la matriz A no es de nxn, debe retornar None
    """
    nops = 0
    m, n = A.shape
    a_1 = A[:, 0]
    r_11 = norma(a_1, 2)
    nops += a_1.shape[0]*2 -1     # operaciones de la norma
    
    q_1 = a_1 / r_11
    nops += 1
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    Q[:, 0] = q_1
    R[0][0] = r_11

    for j in range(1, n):
        q_j_prima = A[:, j] 
        
        R_col_j = multi_matricial(Q[:, :j].T, q_j_prima) 
        q_j_prima = q_j_prima - multi_matricial(Q[:, :j], R_col_j )
        R[:j, j] = R_col_j 
        
        r_jj = norma(q_j_prima, 2)
        
        q_j = q_j_prima / r_jj
        
        Q[:, j] = q_j
        R[j, j] = r_jj
    

    R_economica = R[:n, :] 
    Q_economica = Q[:, :n]

    if (retorna_nops):
        return Q_economica, R_economica, nops

    
    return Q_economica, R_economica

def QR_con_HH(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de mxn (m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m >= n, debe retornar None
    """
    m, n = A.shape
    if (m < n): return None, None

    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        x = R[k:, k]
        alpha = -np.sign(x[0])*norma(x, 2)
        e1 = np.zeros(x.shape[0])
        e1[0] = 1
        u = x- alpha * e1
        normaU = norma(u, 2)
        if (normaU > tol):
            u = u / normaU
            Hk = np.eye(x.shape[0]) - 2*multi_matricial(traspuesta(u), u)
            Hk_prima = np.eye(m)
            Hk_prima[k:, k:] = Hk
            R = multi_matricial(Hk_prima, R)
            Q = multi_matricial(Q, traspuesta(Hk_prima))
    
    return Q, R

def calculaQR(A, metodo='RH', tol=1e-12):
    """
    A una matriz de nxn
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectorers de Householder (RH) o GramSchmidt (GS)
    para realizar la factorizacion.
    retorna matrices Q y R calculadas con RH o GS (y como tercer argumento 
    opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if (metodo == 'GS'):
        return QR_con_GS(A, tol)
    else:
        return QR_con_HH(A, tol)

## Laboratorio 6

def f_A(A, v, k):
    w = v
    for i in range (k):
        w = multi_matricial(A, w)
        norma_2 = norma(w, 2)
        if (norma_2 <= 0): return np.zeros(v.shape[0]) 
        w = w/norma_2
    return w

def metpot2k(A, tol=1e-15, K=1000):
    """
    A una matriz de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del
    autovector.
    K el numero maximo de iteraciones a realizarse.
    Return vector v, autovalor lambda y numero de iteracion realizadas k
    """
    v = np.random.rand(A.shape[0])
    v_prima = f_A(A, v, 2)
    e = multi_matricial(v_prima, traspuesta(v))
    k = 0
    while (abs(e-1)>tol and k < K):
        v = v_prima
        v_prima = f_A(A, v, 1)
        e = multi_matricial(v_prima, traspuesta(v))
        k += 1
    autovalor = multi_matricial(v_prima, traspuesta(multi_matricial(A, v_prima)))
    e -= 1
    return v, autovalor, k

def diagRH(A,tol=1e-15,K=1000):
    
    """
    A una matriz simetrica de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del
    autovector. K el numero maximo de iteraciones a realizarse
    return a matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T
    Si la matriz A no es simetrica, debe retornar None
    """
    n = A.shape[0]
    v1, lambda1, k = metpot2k(A, tol, K)
    e1 = np.zeros(v1.shape[0])
    e1[0] = 1

    sign = 1.0 if v1[0] >= 0 else -1.0
    w = v1 + sign * norma(v1, 2) * e1

    nw = norma(w, 2)
    if (nw < tol):
        H_v1 = np.eye(n)
    else:
        w = w / nw
        H_v1 = np.eye(n) - 2.0 * multi_matricial(traspuesta(w), w)

    if (n == 1):
        S = H_v1
        D = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        return S, D
    
    else:
        B = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        A_prima = B[1:,1:]
        S_prima, D_prima = diagRH(A_prima, tol, K)
        D = np.zeros((n, n))
        D[0][0] = lambda1
        D[1:,1:] = D_prima
        S = np.eye(n)
        S[1:,1:] = S_prima
        S = multi_matricial(H_v1, S)
        return S, D

## Laboratorio 7

def matriz_al_azar_valores_entre_0_y_1(n):
    '''Retorna una matriz aleatoria de NxN entradas'''
    A = np.zeros((n, n))
    for i in range (n):
        for j in range(n):
            A[i][j] = np.random.uniform(0, 1)
    return A

def normalizar_columnas(A, p):
    '''Normaliza la columna dada (con norma p)'''
    A = traspuesta(A)
    A = normaliza(A, p)
    return traspuesta(A)

def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    if (n == 1):
        return np.eye(n)
    A = matriz_al_azar_valores_entre_0_y_1(n)
    return normalizar_columnas(A, 1)
    
def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    A = matriz_al_azar_valores_entre_0_y_1(n)
    for i in range(n):
        for j in range(n):
            if (A[i][j] <= thres):
                A[i][j] = 1
            else:
                A[i][j] = 0
    for i in range(n):
        if (norma(A[:, i], 1) == 0):
            A[np.random.randint(n)][i] = 1
    return normalizar_columnas(A, 1)
    
def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    A_prima = multi_matricial(traspuesta(A), A)
    S, D = diagRH(A_prima, tol, K=1e5)
    autovalores = np.diag(D)
    
    indices_nucleo = []
    for i in range(len(autovalores)):
        if np.abs(autovalores[i]) <= tol:
            indices_nucleo.append(i)
    if (indices_nucleo == []):
        return np.array([])
    base_nucleo = S[:, indices_nucleo]
    return base_nucleo

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    dimensiones = (m_filas, n_columnas)

    if (listado == []):
        return {}, dimensiones
    
    indices_i = listado[0]
    indices_j = listado[1]
    valores_A_ij = listado[2]
    rala = {}
    num_elementos = len(indices_i)
    
    for k in range(num_elementos):
        
        valor = valores_A_ij[k]
        if np.abs(valor) >= tol:
            i = indices_i[k]
            j = indices_j[k]
            rala[(i, j)] = valor
            
    return rala, dimensiones

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    rala = A[0]
    m_filas, n_columnas = A[1]
    
    w = np.zeros(m_filas)
    
    for (i, j), valor_A_ij in rala.items():
        w[i] += valor_A_ij * v[j]
        
    return w

## Laboratorio 8

def svd_reducida(A,k="max",tol=1e-15):
    '''A la matriz de interes(de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k) '''
    
    m,n = A.shape
    #Optimizacion, si m < n calculamos primero U. 
    if m < n:
        A = traspuesta(A)
    
    U, diagonal_autovalores, V = calculoSVDReducida(A, tol)
    
    #Si m<n, se calculó primero U, que en la SVD de At toma el lugar de V (At = V Et Ut).
    #Swapeamos entonces para arreglar.
    if m < n:
        U, V = V, U        
    
    vector_epsilon = matriz_diagonal_a_vector(diagonal_autovalores)
    
    if not k == "max":
        U, vector_epsilon, V = retenerValoresSingulares(U, vector_epsilon, V, k)
    
    return U, vector_epsilon, V 

def retenerValoresSingulares(U, vector_epsilon, V, k):
    U_k = U[:, :k]
    V_k = V[:, :k]
    eps_k = vector_epsilon[:k]
    
    return U_k, eps_k, V_k

def calculoSVDReducida(A, tol):
    A_t_A = multi_matricial(traspuesta(A), A)
    
    V, diagonal_autovalores = diagRH(A_t_A)
    
    epsilon_hat, V_hat = reducirMatrices(V, diagonal_autovalores, tol)
    
    U_hat = calcularMatriz(A, V_hat, epsilon_hat)
    
    return U_hat, epsilon_hat, V_hat

def reducirMatrices(A, diagonal_autovalores, tol):
    for i in range(diagonal_autovalores.shape[0]):
        #if diagonal_autovalores[i][i] != 0
        if np.abs(diagonal_autovalores[i][i]) > tol: 
            diagonal_autovalores[i][i] = math.sqrt(diagonal_autovalores[i][i])
        else:
            epsilon_hat = diagonal_autovalores[:i, :i]
            A_hat = A[:,:i]
            return epsilon_hat, A_hat
    
    return diagonal_autovalores, A

def matriz_diagonal_a_vector(diagonal_autovalores):
    """Convierte una matriz diagonal a un array de numpy"""
    vector_epsilon = []
    
    for i in range(diagonal_autovalores.shape[0]):
        vector_epsilon.append(diagonal_autovalores[i][i])

    vector_epsilon = np.array(vector_epsilon)
    
    return vector_epsilon

def calcularMatriz(A,B,diagonal_autovalores):
    """Devuelve la matriz faltante para SVD"""
    matriz_faltante = multi_matricial(A, B) 
    for j in range(matriz_faltante.shape[1]):
        for i in range(matriz_faltante.shape[0]):
            matriz_faltante[i][j] = matriz_faltante[i][j] / diagonal_autovalores[j][j]
    return matriz_faltante

## Funciones necesarias para el Trabajo Practico

def cholesky(A):
    '''Dada una matriz A, devuelve la matriz L de cholesky asociada a esta'''
    
    if not(esSDP(A)):
        raise Exception("Para calcular la factorizacion de Cholesky de una matriz, es necesario que esta sea SDP")
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for j in range(n):
        vector_L_j = L[j, :j] 
        suma_diag = multi_matricial(vector_L_j, vector_L_j)
        valor_raiz = A[j, j] - suma_diag
        L[j, j] = math.sqrt(valor_raiz)
        for i in range(j + 1, n):
            vector_L_i = L[i, :j]
            suma_no_diag = multi_matricial(vector_L_i, vector_L_j)
            L[i, j] = (A[i, j] - suma_no_diag) / L[j, j]

    return L

def esDiagonal(A):
    '''Chequea que la matriz dada sea diagonal'''
    n, m = A.shape
    if n != m:
        return False

    for i in range(n):
        for j in range(i+1, n):
            if A[i][j] != 0 or A[j][i] != 0:
                return False
    return True

def inversaDeMatrizDiagonal(A):
    '''Calcula la inversa de una matriz diagonal dada'''
    if (not esDiagonal(A)): 
        print('La matriz no es diagonal')

    for i in range (A.shape[0]):
        A[i][i] = 1/A[i][i]
   
    return A

def deVectorAMatrizInversa(VecEpsilon):
    '''dado un vector que representa una matriz diagonal, convierte el vector en matriz y calcula su inversa'''
    N = VecEpsilon.shape[0]
    matrizEpsilon = np.zeros((N, N))
    for i in range(N):
        matrizEpsilon[i][i] = 1/VecEpsilon[i]
    
    return matrizEpsilon

def calcularWconQR(Q, R, Y):
    '''Dada Q y R (factorizacion QR de una matriz) e Y, retorna la matriz de pesos a traves de las matrices Q, R y Y 
    (calculando la pseudoinversa con Q y R)'''
    # Tengo que resolver V @ R^T = X (Con V = pseudo-inversa de X)
    # Para usar res_tri tengo que resolver a derecha asi que aplico traspuesta a ambos lados y resuelvo R @ V^T = Q^T
    n = R.shape[0]
    p = Q.shape[0]
    Q_t = traspuesta(Q)
    V_t = np.zeros((n, p))

    # Calculo V^T
    for i in range(p):
        V_t[:, i] = res_tri(R, Q_t[:, i], False)
        
    V = traspuesta(V_t)
    
    # Obtengo W
    W = multi_matricial(Y, V)
    return W, V

def generarMatrizDeConfusion(W, X, Y):
    """
    Genera una matriz de confusion C 2x2 dada una matriz de pesos W, una matriz de embeddings X y una matriz de targets Y
    
    C[0][0] es TP: Imagenes de gatos predichas correctamente como gatos
    
    C[0][1] es FN: Imagenes de gatos predichas como perros
    
    C[1][0] es FP: Imagenes de perros predichas como gatos
    
    C[1][1] es TN: Imagenes de perros predichas correctamente como perros
    """
    
    C = np.zeros((2,2))
    
    results = multi_matricial(W, X)
    
    for i in range(results.shape[1]):
        if results[0][i] > results[1][i]:
            max_arg = 0
        else:
            max_arg = 1
        
        #Si la prediccion fue gato
        if max_arg == 0:
            #Si era un gato
            if Y[0][i] == 1:
                C[0][0] = C[0][0] + 1
            #Si era un perro
            else:
                C[1][0] = C[1][0] + 1
        #Si la prediccion fue perro
        else:
            #Si era un gato
            if Y[0][i] == 1:
                C[0][1] = C[0][1] + 1
            #Si era un perro
            else:
                C[1][1] = C[1][1] + 1
                
    return C

def extraerPorcentajes(C):
    """Dada una matriz de confusion C, extrae los porcentajes de True Positive, False Positive, True Negative y False Negative"""
    C00, C01 = C[0,0], C[0,1]
    C10, C11 = C[1,0], C[1,1]

    TP = C11
    FP = C01
    FN = C10
    TN = C00
    total = C.sum()

    return TP/total, FP/total, TN/total, FN/total

def generarGraficos(matriz_de_confusion_EN, tiempo_EN, 
                    matriz_de_confusion_SVD, tiempo_SVD, 
                    matriz_de_confusion_WQRHH, tiempo_QRHH,
                    matriz_de_confusion_WQRGS, tiempo_QRGS):
    '''genera los graficos de eficiencia y eficacia de los diferentes metodos'''
    
    #Graficos
    metodos = ["EN", "SVD", "QR-HH", "QR-GS"]
    TP_vals = []
    FP_vals = []
    TN_vals = []
    FN_vals = []

    #Tiempos
    tiempos = [tiempo_EN, tiempo_SVD, tiempo_QRHH, tiempo_QRGS]
    plt.figure()
    plt.bar(metodos, tiempos)
    plt.ylabel("Tiempo en segundos")
    plt.title("Comparativa de tiempos de ejecucion para el calculo de W")
    plt.tight_layout()
    plt.show()
    
    for C in [matriz_de_confusion_EN, matriz_de_confusion_SVD, matriz_de_confusion_WQRHH, matriz_de_confusion_WQRGS]:
        TP, FP, TN, FN = extraerPorcentajes(C)
        TP_vals.append(TP)
        FP_vals.append(FP)
        TN_vals.append(TN)
        FN_vals.append(FN)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    for i, metodo in enumerate(metodos):
        #Posicion en el grafico
        ax = axs[i // 2, i % 2] 
        
        values = [TP_vals[i], FP_vals[i], TN_vals[i], FN_vals[i]]
        
        ax.bar(['TP', 'FP', 'TN', 'FN'], values, color=['skyblue', 'lightcoral', 'lightgreen', 'salmon'])
        
        ax.set_title(f'Método {metodo}')
        ax.set_ylabel('Porcentaje')
    
    plt.tight_layout()

    plt.show()

def obtenerMatricesDeConfusion(Xt, Yt, Xv, Yv):
    '''
    Dado una matriz de embeddings de entrenamiento Xt, su target Yt, y una matriz de embeddings de validacion Xv con sus targets Yv,
    genera 4 matrices de confusion con los metodos de Ecuaciones Normales, SVD, QR con Householder y QR con Gram-Schmidt.
    Ademas, devuelve el tiempo que tardo en obtenerse la matriz de pesos para cada metodo.
    '''
    
    # En el contexto del TP n < p, entonces para el algoritmo 1 aplicamos Cholesky sobre X @ X^T
    
    print('Inició WEN')
    tiempo_inicio_EN = time.perf_counter()
    L = cholesky(Xt @ traspuesta(Xt))
    WEN, pXt_EN = pinvEcuacionesNormales(Xt, L , Yt)
    tiempo_fin_EN = time.perf_counter()
    print('Terminó WEN')
    # Verificamos si es pseudo-inversa
    print('Es pseudo-inversa con EN: ', esPseudoInversa(Xt, pXt_EN))
    matriz_de_confusion_EN = generarMatrizDeConfusion(WEN, Xv, Yv)
    
    print('Inició WSVD')
    tiempo_inicio_SVD = time.perf_counter()
    U, s_vector, V = svd_reducida_optimizado(Xt)
    S = np.diag(s_vector)
    WSVD,  pXt_SVD= pinvSVD(U, S, V, Yt)
    tiempo_fin_SVD = time.perf_counter()
    print('Terminó WSVD')
    # Verificamos si es pseudo-inversa
    print('Es pseudo-inversa con SVD: ', esPseudoInversa(Xt, pXt_SVD))
    matriz_de_confusion_SVD = generarMatrizDeConfusion(WSVD, Xv, Yv)

    print('Inició WQRHH')
    tiempo_inicio_QRHH = time.perf_counter()
    QHH, RHH = QR_con_HH_optimizado(traspuesta(Xt))
    WQRHH, pXt_QRHH = pinvHouseHolder(QHH, RHH, Yt)
    tiempo_fin_QRHH = time.perf_counter()
    print('Terminó WQRHH')
    # Verificamos si es pseudo-inversa
    print('Es pseudo-inversa con QRHH: ', esPseudoInversa(Xt, pXt_QRHH))
    matriz_de_confusion_WQRHH = generarMatrizDeConfusion(WQRHH, Xv, Yv)
    
    print('Inició WQRGS')
    tiempo_inicio_QRGS = time.perf_counter()
    QGS, RGS = QR_con_GS(traspuesta(Xt))
    WQRGS, pXt_QRGS = pinvGramSchmidt(QGS, RGS, Yt)
    tiempo_fin_QRGS = time.perf_counter()
    print('Terminó WQRGS')
    # Verificamos si es pseudo-inversa
    print('Es pseudo-inversa con QRGS: ', esPseudoInversa(Xt, pXt_QRGS))
    matriz_de_confusion_WQRGS = generarMatrizDeConfusion(WQRGS, Xv, Yv)

    tiempo_EN = tiempo_fin_EN - tiempo_inicio_EN
    tiempo_SVD = tiempo_fin_SVD - tiempo_inicio_SVD
    tiempo_QRHH = tiempo_fin_QRHH - tiempo_inicio_QRHH
    tiempo_QRGS = tiempo_fin_QRGS - tiempo_inicio_QRGS
    
    return matriz_de_confusion_EN,matriz_de_confusion_SVD,matriz_de_confusion_WQRHH,matriz_de_confusion_WQRGS,tiempo_EN,tiempo_SVD,tiempo_QRHH,tiempo_QRGS

## Soluciones a los ejercicios del TP

# Ejercicio 1
def cargarDataset(carpeta):
    '''Dada una carpeta, carga los datos de esta en diferentes matrices segun corresponda'''
    # El input carpeta debe ser la ruta completa a la carpeta cats_and_dogs
    # carpetas
    train_cats = os.path.join(carpeta, "train", "cats", "efficientnet_b3_embeddings.npy")
    train_dogs = os.path.join(carpeta, "train", "dogs", "efficientnet_b3_embeddings.npy")
    val_cats   = os.path.join(carpeta, "val", "cats", "efficientnet_b3_embeddings.npy")
    val_dogs   = os.path.join(carpeta, "val", "dogs", "efficientnet_b3_embeddings.npy")

    Xtc = np.load(train_cats)   # gatos train
    Xtd = np.load(train_dogs)   # perros train
    Xvc = np.load(val_cats)     # gatos val
    Xvd = np.load(val_dogs)     # perros val

    Xt = np.concatenate((Xtc, Xtd), axis=1)
    Xv = np.concatenate((Xvc, Xvd), axis=1)

    Nc_train = Xtc.shape[1]   
    Nd_train = Xtd.shape[1]  

    Yt_cats = np.zeros((2, Nc_train))
    Yt_dogs = np.zeros((2, Nd_train))

    for i in range(Nc_train):
        Yt_cats[0, i] = 1
        Yt_cats[1, i] = 0

    for i in range(Nd_train):
        Yt_dogs[0, i] = 0
        Yt_dogs[1, i] = 1

    Yt = np.concatenate((Yt_cats, Yt_dogs), axis=1)

    Nc_val = Xvc.shape[1]  
    Nd_val = Xvd.shape[1]  
    
    Yv_cats = np.zeros((2, Nc_val))
    Yv_dogs = np.zeros((2, Nd_val))

    for i in range(Nc_val):
        Yv_cats[0, i] = 1
        Yv_cats[1, i] = 0

    for i in range(Nd_val):
        Yv_dogs[0, i] = 0
        Yv_dogs[1, i] = 1

    Yv = np.concatenate((Yv_cats, Yv_dogs), axis=1)

    return Xt, Yt, Xv, Yv

# Ejercicio 2 
def pinvEcuacionesNormales(X,L,Y):
    '''Dadas X la matriz de embeddings 
    L la matriz de Cholesky
    Y la matriz de targets de entrenamiento
    Calcula la matriz de pesos W utilizando las ecuaciones normales para la resolucion de la pseudoinversa'''
    n, p = X.shape
    X_t = traspuesta(X)
    L_t = traspuesta(L)
    if n > p:
        # Asumo que L = cholesky(X^T @ X)
        # Quiero resolver L @ L^T @ U = X^T 
        Z = np.zeros((p, n))
        U = np.zeros((p, n))
        
        # Paso intermedio. Sustitucion hacia adelante: L @ Z = X^T
        for i in range(n):
            Z[:, i] = res_tri(L, X_t[:, i], True)
            
        # Resuelvo el sistema. Sustitución hacia atrás: L^T @ U = Z
        for i in range(n):
            U[:, i] = res_tri(L_t, Z[:, i], False)
        
        V = U
        # Calculo W
        W = multi_matricial(Y, V)
    
    elif n < p:
        # Asumo que L = cholesky(X @ X^T)
        # Quiero resolver V @ X @ X^T = X^T
        # Para usar res_tri tengo que resolver a derecha asi que aplico traspuesta a ambos lados y resuelvo L @ L^T @ V^T = X
        
        Z = np.zeros((n, p))
        Vt = np.zeros((n, p))

        # Paso intermedio. Sustitucion hacia adelante: L @ Z = X
        for i in range(p):
            Z[:, i] = res_tri(L, X[:, i], True)
            
        # Resuelvo el sistema. Sustitución hacia atrás: L^T @ V^T = Z
        for i in range(p):
            Vt[:, i] = res_tri(L_t, Z[:, i], False)
        
        V = traspuesta(Vt)
        # Calculo W
        W = multi_matricial(Y, V)
        
    else:
        # Como la pseudoinversa X^+ = X^-1 entonces W = Y @ X^-1
        
        V = inversa(X)
        # Calculo W
        W = multi_matricial(Y, V)
    return W, V

# Ejercicio 3
def pinvSVD(U, S, V, Y):
    '''Dadas las maatrices U, S y V de la descomposicion SVD e Y la matriz de Targets de entranamiento, calcula la matriz de pesos W utilizando la descomposicion SVD para la resolucion de la pseudoinversa'''
    
    n = S.shape[0]

    # Calculamos Sigma_1^-1
    S_1 = inversaDeMatrizDiagonal(S[:, :n])

    # Calculamos la pseudo-inversa de X
    V_1 = V[:,:n]
    U_1 = U[:,:n]
    pseudoInversa = multi_matricial(multi_matricial(V_1, S_1), traspuesta(U_1))
    
    W = multi_matricial(Y, pseudoInversa)

    return W, pseudoInversa

# Ejercicio 4
def pinvHouseHolder(Q,R,Y):
    '''Dadas las matrices Q,R de la descomposicion QR utilizando HouseHolder e Y la matriz de targets de entrenamiento. La funcion devuelve la matriz de pesos W utilizando la factorizacion QR para la resolucion de la pseudoinversa.'''
    W, V = calcularWconQR(Q, R, Y)
    return W, V

def pinvGramSchmidt(Q,R,Y):
    '''Dadas las matrices Q,R de la descomposicion QR utilizando GramSchimdt e Y la matriz de targets de entrenamiento. La funcion devuelve la matriz de pesos W utilizando la factorizacion QR para la resolucion de la pseudoinversa.'''
    W, V = calcularWconQR(Q, R, Y)
    return W, V

# Ejercicio 5
def esPseudoInversa(X, pX, tol = 1e-8):
    '''Chequeo si dadas dos matrices, X y pX, pX es la pseudoinversa de X'''
    #La pseudo inversa es la unica matriz que cumple los 4 puntos mencionados en el tp (al final de la pagina 3)
    # 1) X pX X = X
    if not matricesIguales(multi_matricial(X, multi_matricial(pX, X)), X, tol):
        return False
    # 2) pX X pX = pX
    if not matricesIguales(multi_matricial(pX, multi_matricial(X, pX)), pX, tol):
        return False
    # 3) (X pX)^T = X pX
    XpX = multi_matricial(X, pX)
    if not matricesIguales(traspuesta(XpX), XpX, tol):
        return False
    # 4) (pX X)^T = pX X
    pXX = multi_matricial(pX, X)
    if not matricesIguales(traspuesta(pXX), pXX, tol):
        return False
    return True

## Funciones optimizadas para poder correr el TP en un tiempo razonable:

def svd_reducida_optimizado(A,k="max",tol=1e-15):
    '''
    Esta funcion no tiene ninguna optimizacion, solo llama a calculoSVDReducida_optimizado
    '''
    
    m,n = A.shape
    #Si m < n calculamos primero U. 
    if m < n:
        A = traspuesta(A)
    
    U, diagonal_autovalores, V = calculoSVDReducida_optimizado(A, tol)
    if m < n:
        U, V = V, U        
    
    vector_epsilon = matriz_diagonal_a_vector(diagonal_autovalores)
    
    if not k == "max":
        U, vector_epsilon, V = retenerValoresSingulares(U, vector_epsilon, V, k)
    
    return U, vector_epsilon, V 

def calculoSVDReducida_optimizado(A, tol):
    '''Calculo de la factorizacion SVD de A utilizando eigh para mayor velocidad'''
    A_t_A = multi_matricial(traspuesta(A), A)
    
    A_t_A = multi_matricial(traspuesta(A), A)
    
    w, V = eigh(A_t_A)
    diagonal_autovalores = np.diag(w)

    epsilon_hat, V_hat = reducirMatrices(V, diagonal_autovalores, tol)
    
    U_hat = calcularMatriz(A, V_hat, epsilon_hat)
    
    return U_hat, epsilon_hat, V_hat

'''
En cuanto al calculo de QR con HH, tardaba muchisimo (horas) con nuestra implementacion de multi_matricial. Simplemente reemplazamos esas por @
'''

def QR_con_HH_optimizado(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de mxn (m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m >= n, debe retornar None
    """
    m, n = A.shape
    if (m < n): return None, None

    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        x = R[k:, k]
        alpha = -np.sign(x[0])*norma(x, 2)
        e1 = np.zeros(x.shape[0])
        e1[0] = 1
        u = x- alpha * e1
        normaU = norma(u, 2)
        if normaU > tol:
            u /= normaU

            # OPTIMIZACION: plicar la transformación de Householder directamente a las matrices R y Q sin formar H_k, vectorizanco 
            # actualizao R
            uR = u @ R[k:, k:]
            u = u.reshape(-1, 1)   
            uR = uR.reshape(1, -1)
            R[k:, k:] -= 2.0 * (u @ uR)

            # actualizo Q
            Qu = Q[:, k:] @ u
            Qu = Qu.reshape(-1, 1)   
            u = u.reshape(1, -1)
            Q[:, k:] -= 2.0 * (Qu @ u)

    R_economica = R[:n, :] 
    Q_economica = Q[:, :n]
    
    return Q_economica, R_economica

## Funciones extra para calcular, comparar y/o estimar tiempos de ejecucion. rt por execution time

def QR_con_HH_sin_arroba(A, numeroDeIteraciones, tol=1e-12):
    '''
    Esta funcion realiza las primeras iteraciones de QR_con_HH, utilizando nuestra implementacion de multi_matricial.
    '''
    m, n = A.shape
    if (m < n): return None, None

    R = A.copy()
    Q = np.eye(m)

    for k in range(numeroDeIteraciones):
        x = R[k:, k]
        alpha = -np.sign(x[0])*norma(x, 2)
        e1 = np.zeros(x.shape[0])
        e1[0] = 1
        u = x- alpha * e1
        normaU = norma(u, 2)
        if normaU > tol:
            u /= normaU

            # OPTIMIZACION: plicar la transformación de Householder directamente a las matrices R y Q sin formar H_k, vectorizanco 
            # actualizao R
            uR = multi_matricial(u, R[k:, k:])
            u = u.reshape(-1, 1)   
            uR = uR.reshape(1, -1)
            R[k:, k:] -= 2.0 * multi_matricial(u, uR)

            # actualizo Q
            Qu = multi_matricial(Q[:, k:], u)
            Qu = Qu.reshape(-1, 1)   
            u = u.reshape(1, -1)
            Q[:, k:] -= 2.0 * multi_matricial(Qu, u)

    R_economica = R[:n, :] 
    Q_economica = Q[:, :n]
    
    return 

def diagRH_sin_arroba(A, K, numeroDeRecursiones,tol=1e-15):
    '''
    Esta funcion realiza los primeros pasos recursivos de diagRH, utilizando nuestra implementacion de multi_matricial.
    Debido a que metpot2k no cambia significativamente el tiempo de ejecicion utilizando @ en lugar de nuestra multi_matricial,
    optamos por dejar tanto metpot2k como f_A con multi_matricial.
    Esta funcion permite variar tanto el K, es decir la cantidad de iteraciones para el metodo de la potencia como la cantidad de pasos
    recursivos a realizar.
    '''
    if numeroDeRecursiones == 0:
        return
    n = A.shape[0]
    v1, lambda1, k = metpot2k(A, tol, K)
    e1 = np.zeros(v1.shape[0])
    e1[0] = 1

    sign = 1.0 if v1[0] >= 0 else -1.0
    w = v1 + sign * norma(v1, 2) * e1

    nw = norma(w, 2)
    if (nw < tol):
        H_v1 = np.eye(n)
    else:
        w = w / nw
        H_v1 = np.eye(n) - 2.0 * multi_matricial(traspuesta(w), w)

    if (n == 1):
        S = H_v1
        D = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        return
    
    else:
        B = multi_matricial(H_v1, multi_matricial(A, traspuesta(H_v1)))
        A_prima = B[1:,1:]
        diagRH_sin_arroba(A_prima, K, numeroDeRecursiones-1,tol=1e-15)
        # Aca estamos incluso ahorrando una multiplicacion matricial correspondiente a S = multi_matricial(H_v1, S)
        return

def proyeccion_y_rt_QR_con_HH_sin_arroba(Xt, i):
    A = traspuesta(Xt)
    tiempo_inicio = time.perf_counter()
    QR_con_HH_sin_arroba(A, i)
    tiempo_fin = time.perf_counter()
    runtime = tiempo_fin - tiempo_inicio
    proyeccion = (A.shape[1]/i)*runtime
    return runtime, proyeccion

def proyeccion_y_rt_diagRH_sin_arroba(Xt, K, numeroDeRecursiones):
    A = (Xt @ (Xt).T)
    tiempo_inicio = time.perf_counter()
    diagRH_sin_arroba((A), K, numeroDeRecursiones)
    tiempo_fin = time.perf_counter()
    runtime = tiempo_fin - tiempo_inicio
    proyeccion = ((Xt).shape[0]/numeroDeRecursiones)*runtime*2 #Este *2 se debe a que no se esta calculando lo que tarda la vuelta de la recursion y que tiene que hacer tmb multiplicaciones matriciales sin @
    return runtime, proyeccion

# Funciones para calcular runtime de calcular W con SVD usando diagRH con @
'''
En cuanto a el calculo de svd, logramos hacer que diagRH corriera en un tiempo razonable, utilizando @ y K = 1 para el metodo de la potencia.
Sin embargo surgieron otros problemas. Uno es que la matriz es de 1536x1536, la recursion de python llega hasta poco menos de 1000.
Tuvimos que cambiar el limite de las recursiones a 2000 para tener un poco de margen y que ande.
No voy a negar que al hacer eso me crasheo la compu porque se quedo sin memoria y tuve que forzar un apagado jajaja.
Asi que tambien tuvimos que reducir el gasto de memoria, convirtiendo todo a float16 (consume unos 9GB de RAM) 
'''
dtype = np.float16
import sys
N_MAXIMO_RECURSION = 2000 
sys.setrecursionlimit(N_MAXIMO_RECURSION)

def diagRH_con_arroba(A, tol=1e-15,K=1):
    """
    Esta es la unica funcion optimizada en el calculo de SVD. Unico cambio utilizar @ en lugar de multi_matricial
    """
    # Convertimos la entrada A a float16 para ahorrar memoria (si no se corta por RecursionError: maximum recursion depth exceeded)
    if A.dtype != dtype:
        A = A.astype(dtype)
    n = A.shape[0]
    print(n)
    v1, lambda1, k = metpot2k(A, tol, K)
    v1 = v1.astype(dtype)
    e1 = np.zeros(v1.shape[0], dtype=dtype)
    e1[0] = 1

    sign = 1.0 if v1[0] >= 0 else -1.0
    w = v1 + sign * norma(v1, 2) * e1

    nw = norma(w, 2)
    if (nw < tol):
        H_v1 = np.eye(n, dtype=dtype)
    else:
        w = w / nw
        H_v1 = np.eye(n, dtype=dtype) - dtype(2.0) * (traspuesta(w) @ w.reshape(1, -1))

    if (n == 1):
        S = H_v1
        D = (H_v1 @ (A @ traspuesta(H_v1)))
        return S, D
    
    else:
        B = (H_v1 @ (A @ traspuesta(H_v1))).astype(dtype)
        A_prima = B[1:,1:]
        S_prima, D_prima = diagRH_con_arroba(A_prima, tol, K)
        print(n)
        D = np.zeros((n, n), dtype=dtype)
        D[0][0] = lambda1
        D[1:,1:] = D_prima
        S = np.eye(n, dtype=dtype)
        S[1:,1:] = S_prima
        S = (H_v1 @ S)
        return S, D

def svd_reducida_con_arroba(A,k="max",tol=1e-15):
    '''
    Esta funcion no tiene ninguna optimizacion, solo llama a calculoSVDReducida_optimizado
    '''
    
    m,n = A.shape
    if m < n:
        A = traspuesta(A)
    
    U, diagonal_autovalores, V = calculoSVDReducida_optimizado(A, tol)
    if m < n:
        U, V = V, U        
    
    vector_epsilon = matriz_diagonal_a_vector(diagonal_autovalores)
    
    if not k == "max":
        U, vector_epsilon, V = retenerValoresSingulares(U, vector_epsilon, V, k)
    
    return U, vector_epsilon, V 

def calculoSVDReducida_con_arroba(A, tol):
    '''
    Esta funcion no tiene ninguna optimizacion, solo llama a diagRH_con_arroba
    '''
    A_t_A = (traspuesta(A) @ A)
    
    diagonal_autovalores, V = diagRH_con_arroba(A_t_A)

    epsilon_hat, V_hat = reducirMatrices(V, diagonal_autovalores, tol)
    
    U_hat = calcularMatriz(A, V_hat, epsilon_hat)
    
    return U_hat, epsilon_hat, V_hat

def rt_calcular_W_usando_diagRH_con_arroba(Xt, K, numeroDeRecursiones):
    A = (Xt @ (Xt).T)
    tiempo_inicio = time.perf_counter()
    diagRH_sin_arroba((A), K, numeroDeRecursiones)
    tiempo_fin = time.perf_counter()
    runtime = tiempo_fin - tiempo_inicio
    proyeccion = ((Xt).shape[0]/numeroDeRecursiones)*runtime*2 #Este *2 se debe a que no se esta calculando lo que tarda la vuelta de la recursion y que tiene que hacer tmb multiplicaciones matriciales sin @
    return runtime, proyeccion




# Funcion para convertir segundos a minutos/horas/dias

def formatear_tiempo(proyeccion: float) -> str:
    """
    Esto estuvo hecho con GPT :p
    Convierte una cantidad de segundos en un string legible,
    mostrando solo las dos unidades de tiempo más grandes (días, horas, minutos, segundos)
    que sean distintas de cero.

    :param proyeccion: Cantidad de segundos (puede ser float) a formatear.
    :return: String con el formato 'X unidad_mayor y Y unidad_menor'.
    """

    # Definición de unidades en segundos
    SEGUNDOS_EN_DIA = 86400
    SEGUNDOS_EN_HORA = 3600
    SEGUNDOS_EN_MINUTO = 60

    # 1. Aseguramos que la entrada es un entero para los cálculos modulares
    proyeccion_entero = int(round(proyeccion))
    tiempo_restante = proyeccion_entero

    # 2. Cálculos para obtener cada unidad
    dias = tiempo_restante // SEGUNDOS_EN_DIA
    tiempo_restante %= SEGUNDOS_EN_DIA

    horas = tiempo_restante // SEGUNDOS_EN_HORA
    tiempo_restante %= SEGUNDOS_EN_HORA

    minutos = tiempo_restante // SEGUNDOS_EN_MINUTO
    segundos = tiempo_restante % SEGUNDOS_EN_MINUTO

    # 3. Lista de unidades (valor, nombre_singular, nombre_plural)
    unidades = [
        (dias, "día", "días"),
        (horas, "hora", "horas"),
        (minutos, "minuto", "minutos"),
        (segundos, "segundo", "segundos"),
    ]

    # 4. Filtramos solo las unidades con valor > 0
    unidades_significativas = [(v, s, p) for v, s, p in unidades if v > 0]

    # 5. Lógica de formato (tomando solo las 2 más grandes)

    if not unidades_significativas:
        # Caso de 0 segundos
        return "0 segundos"
    
    # Tomamos las dos unidades más grandes
    unidad_mayor = unidades_significativas[0]
    
    # Formateo de la unidad mayor (incluye singular/plural)
    def formatear_unidad(valor, singular, plural):
        nombre = singular if valor == 1 else plural
        return f"{valor} {nombre}"

    resultado = formatear_unidad(unidad_mayor[0], unidad_mayor[1], unidad_mayor[2])

    # Si hay una segunda unidad, la añadimos con "y"
    if len(unidades_significativas) >= 2:
        unidad_menor = unidades_significativas[1]
        resultado += " y " + formatear_unidad(unidad_menor[0], unidad_menor[1], unidad_menor[2])
    
    return resultado
