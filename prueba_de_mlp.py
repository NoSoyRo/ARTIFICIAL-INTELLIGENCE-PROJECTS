import perceptron as pcn
import numpy as np

in_not   = np.array([[0],[1]])
targ_not = np.array([[1],[0]])

in_nand   = np.array([[0,0],[0,1],[1,0],[1,1]])
targ_nand = np.array([[1],[1],[1],[0]])

in_nor   = np.array([[0,0],[0,1],[1,0],[1,1]])
targ_nor = np.array([[1],[0],[0],[0]])

pcnseq = pcn.mlpcn(in_nor,targ_nor,1,tipo_funact = "logistica")
pcnseq.mlp_batch_entrenamiento(in_nor, targ_nor, 0.25, 5000, 100)
a = pcnseq.predicciones

pcnseq = pcn.mlpcn(in_nor,targ_nor,1,tipo_funact = "logistica")
pcnseq.mlp_secuencial_entrenamiento(in_nor, targ_nor, 0.25, 5000, 100)
b = pcnseq.predicciones

print(a,b)
