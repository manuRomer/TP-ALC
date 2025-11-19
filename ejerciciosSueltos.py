from alc import *

carpetaGatosYPerros = '/home/manu/Escritorio/TP alc/TP-ALC/template-alumnos/dataset/cats_and_dogs/'

# Ejercicio 6         
def evaluacion():

    Xt, Yt, Xv, Yv = cargarDataset(carpetaGatosYPerros)

    C_EN, C_SVD, C_WQRHH, C_WQRGS, tiempo_EN, tiempo_SVD, tiempo_QRHH, tiempo_QRGS = obtenerMatricesDeConfusion(Xt, Yt, Xv, Yv)
    
    generarGraficos(C_EN, tiempo_EN, 
                    C_SVD, tiempo_SVD, 
                    C_WQRHH, tiempo_QRHH, 
                    C_WQRGS, tiempo_QRGS)

    
    

evaluacion()