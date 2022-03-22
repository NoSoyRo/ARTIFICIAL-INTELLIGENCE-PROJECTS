import perceptron as pcn
import numpy as np

#se puede jugar con los dos nuevos modelos creados: 
# pcn_trainsecuencial_qsdcnseeep y pcn_trainsecuencial_qsdcnseen
# que son secuenciales que se detienen en cuanto aciertan y uno que se detiene cuando 
# acierta en todas las predicciones, respectivamente.

in_not   = np.array([[0],[1]])
targ_not = np.array([[1],[0]])

in_nand   = np.array([[0,0],[0,1],[1,0],[1,1]])
targ_nand = np.array([[1],[1],[1],[0]])

in_nor   = np.array([[0,0],[0,1],[1,0],[1,1]])
targ_nor = np.array([[1],[0],[0],[0]])

pcnseq = pcn.pcn(in_nor,targ_nor)
pcnseq.pcn_trainsecuencial_qsdcnseeep(in_nor, targ_nor, 0.25, 5)

print(pcnseq.predecir(np.array([0,0])))
print(pcnseq.predecir(np.array([1,0])))
print(pcnseq.predecir(np.array([0,1])))
print(pcnseq.predecir(np.array([1,1])))

##notamos que hay un problema pues inmediatamente se deja de equivocar, deja de corregir sin importar si despues se equivoca.
#así pues existe el segundo modelo que si usa todos pero de modo secuencial.
pcnseq = pcn.pcn(in_nor,targ_nor)
pcnseq.pcn_trainsecuencial_qsdcnseen(in_nor, targ_nor, 0.25, 10)
print(pcnseq.predecir(np.array([0,0])))
print(pcnseq.predecir(np.array([1,0])))
print(pcnseq.predecir(np.array([0,1])))
print(pcnseq.predecir(np.array([1,1])))


