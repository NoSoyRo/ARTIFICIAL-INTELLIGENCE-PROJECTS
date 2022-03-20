import numpy as np

### Algoritmo realizado con base en el algoritmo de la página 47 del libro: "Machine Learnum_entradag: An Algorithmic Perspective de Stephen Marsland"

class pcn:
	""" Básico perceptron"""
	def __init__(self,entrada,resultados):
		if np.ndim(entrada)>1:
			self.num_entrada = np.shape(entrada)[1]
		else: 
			self.num_entrada = 1
		if np.ndim(resultados)>1:
			self.dim_salida = np.shape(resultados)[1]
		else:
			self.dim_salida = 1
		self.num_de_datos = np.shape(entrada)[0]
		self.pesos = np.random.rand(self.num_entrada+1,self.dim_salida)*0.1-0.05
	def pcntrain(self,entrada,resultados,eta,num_iteraciones):
		entrada = np.concatenate((entrada,-np.ones((self.num_de_datos,1))),axis=1)
		for n in range(num_iteraciones):	
			self.activaciones = self.pcn_hacia_delante(entrada);
			self.pesos -= eta*np.dot(np.transpose(entrada),self.activaciones-resultados)
	def pcn_hacia_delante(self,entrada):
		activaciones =  np.dot(entrada,self.pesos)
		return np.where(activaciones>0,1,0)
	def mat_conf(self,entrada,resultados):
		entrada = np.concatenate((entrada,-np.ones((self.num_de_datos,1))),axis=1)
		salidas = np.dot(entrada,self.pesos)
		num_clases = np.shape(resultados)[1]
		if num_clases==1:
			num_clases = 2
			salidas = np.where(salidas>0,1,0)
		else:
			salidas = np.argmax(salidas,1)
			resultados = np.argmax(resultados,1)
		cm = np.zeros((num_clases,num_clases))
		for i in range(num_clases):
			for j in range(num_clases):
				cm[i,j] = np.sum(np.where(salidas==i,1,0)*np.where(resultados==j,1,0))
		print cm
		print np.trace(cm)/np.sum(cm)
