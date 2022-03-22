import numpy as np

##Algoritmos de deep learning.

def normalizacion(datos,targets):
	##axis = 0 suma columnas y axis = 1 suma las filas de una matriz.
	return((datos-datos.mean(axis = 0))/datos.var(axis = 0),(targets-targets.mean(axis = 0))/targets.var(axis = 0))

### Algoritmos realizados con base en los algoritmos del libro: "Machine Learnum_entradag: An Algorithmic Perspective de Stephen Marsland"


def linreg(entradas,targets):

	entradas = np.concatenate((entradas,-np.ones((np.shape(entradas)[0],1))),axis=1)
	beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(entradas),entradas)),np.transpose(entradas)),targets)

	outputs = np.dot(entradas,beta)
	#print np.shape(beta)
	#print outputs
	return beta, outputs

class pcn():
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
			self.activaciones = self.pcn_hacia_delante(entrada)
			self.pesos -= eta*np.dot(np.transpose(entrada),self.activaciones-resultados)
	def pcn_hacia_delante(self,entrada):
		activaciones =  np.dot(entrada,self.pesos)
		return(np.where(activaciones>0,1,0))	
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
	def pcn_trainsecuencial_qsdcnseeep(self,entrada,resultados,eta,num_iteraciones):
		###pcn secuencial que se detiene cuando no se equivoca en el primero.
		entrada = np.concatenate((entrada,-np.ones((self.num_de_datos,1))),axis=1)
		self.equivocacion = True ##significa que asumes que estas iniciando equivocandote. owo
		for n in range(num_iteraciones):
			if self.equivocacion==True:
				for i in range(entrada.shape[0]):
					if self.equivocacion==True:
						vec_pred = np.matmul(entrada[i,:],self.pesos)
						activaciones = np.where(vec_pred>0,1,0) #activaciones de neuronas para el primer dato
						for k in range(activaciones.shape[0]):
							## corrección de los pesos de la k-esima neurona
							pred_aux,corr_aux = np.full((self.pesos.shape[0],),activaciones[k]),np.full((self.pesos.shape[0],),resultados[i,k]) 
							delta_y_i_neurona = pred_aux-corr_aux
							self.pesos[:,k] = self.pesos[:,k] - eta*np.multiply(entrada[i,:],delta_y_i_neurona)
						#cambio de variable de self.equivocacion
						if np.array_equal(activaciones,resultados[i]) == True:
							self.equivocacion = False
							#print("en el siguiente dato:", i, "predijo bien")
					else:
						break
			else:
				break
	def pcn_trainsecuencial_qsdcnseen(self,entrada,resultados,eta,num_iteraciones):
		###pcn secuencial que se detiene cuando no se equivoca en ninguna.
		entrada = np.concatenate((entrada,-np.ones((self.num_de_datos,1))),axis=1)
		self.equivocacion = True ##significa que asumes que estas iniciando equivocandote. owo
		for n in range(num_iteraciones):
			if self.equivocacion==True:
				for i in range(entrada.shape[0]):
					vec_pred = np.matmul(entrada[i,:],self.pesos)
					activaciones = np.where(vec_pred>0,1,0) #activaciones de neuronas para el primer dato
					for k in range(activaciones.shape[0]):
						## corrección de los pesos de la k-esima neurona
						pred_aux,corr_aux = np.full((self.pesos.shape[0],),activaciones[k]),np.full((self.pesos.shape[0],),resultados[i,k]) 
						delta_y_i_neurona = pred_aux-corr_aux
						self.pesos[:,k] = self.pesos[:,k] - eta*np.multiply(entrada[i,:],delta_y_i_neurona)
					#cambio de variable de self.equivocacion
				predicciones = np.dot(entrada,self.pesos)
				activaciones_mat = np.where(predicciones>0,1,0)
				if np.array_equal(activaciones_mat,resultados) == True:
					self.equivocacion = False
					#print("en la iteracion:", n,"predijo bien puesto que: \n", activaciones_mat.transpose(), "y", resultados.transpose(), "son iguales.")
				#print("iteracion", n)
			else:
				break
	def predecir(self, entrada):
		if np.array_equal(entrada.shape ,self.pesos[:,0].shape):
			pred = np.dot(entrada,self.pesos)
		else:
			entrada_corregida = np.append(entrada, -1)
			pred = np.dot(entrada_corregida, self.pesos)
		if pred > 0:
			act = 1
		else:
			act = 0
		return(pred, act)

		

