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

class mlpcn(): #solo de un layer oculto.
	def __init__(self, entradas, targets, n_oc, beta = 1, momentum = 0.9, tipo_funact = "logistica"): #logistic=sigmoidal
		#tamaños
		#numero de características que tiene cada dato: n_caract 
		#numero de nodos ocultos: n_nodoc
		self.n_caract     = np.shape(entradas)[1]
		self.n_datos      = np.shape(entradas)[0]
		self.n_nodoc      = n_oc
		self.n_out        = np.shape(targets)[1]
		self.beta         = beta
		self.momentum     = momentum
		self.tipo_funact  = tipo_funact
		#generamos las matrices de pesos para una red neuronal de una sola capa oculta
		self.pesos1 = (np.random.rand(self.n_caract+1,self.n_nodoc)-0.5)*2/np.sqrt(self.n_caract) #tan cercano como se pueda al 0 para conservar comportameinto lineal.
		self.pesos2 = (np.random.rand(self.n_nodoc+1,self.n_out)-0.5)*2/np.sqrt(self.n_nodoc)     #tan cercano como se pueda al 0 para conservar comportameinto lineal.
	def mlp_batch_entrenamiento(self,entradas, targets, eta, iteraciones, tau, grafica_curva_de_apr = False): 
		#tau es para ir calculando el error cada tau iteraciones *** tambien, 
		#este algoritmo no necesita un early stop pq es segura su convergencia.
		datos         = np.concatenate((entradas, -np.ones((self.n_datos,1))), axis = 1)
		pesos1_nuevos = np.zeros(np.shape(self.pesos1))
		pesos2_nuevos = np.zeros(np.shape(self.pesos2))
		if grafica_curva_de_apr == True:
				dominio     = [i for i in range(iteraciones)]
				error_mse   = []
		for n in range(iteraciones):
			self.predicciones = self.mlp_forward(datos)
			if grafica_curva_de_apr == True:
				error_mse.append(0.5*np.sum((self.predicciones-targets)**2))
			#distintos tipos de dalidas debido a sus distintas funciones de activacion:
			if self.tipo_funact == "lineal":
				res1 = self.predicciones-targets
				#res2 = self.predicciones-targets/self.n_datos <--- la diferencia esta en que aquí divide por el numero total de datos que tenemos, aunque no es necesario ayud a reescalar los datos
				delta_o = res1
			elif self.tipo_funact == "logistica":
				delta_o = self.beta*(self.predicciones-targets)*(self.predicciones)*(1-self.predicciones) #lol te amo numpy
			elif self.tipo_funact == "softmax":
				delta_o = (self.predicciones-targets)*(self.predicciones*(-self.predicciones)+self.predicciones)/self.n_datos 
			else:
				print("aun no implemento esta funcion de activacion")
			#recuerda que delta_o es un arreglo que ya contiene delta_o(k) o sea ya es un arreglo que contiene todos los deltas_o de k
			#calculamos el delta hidden:
			delta_h = self.beta*self.activaciones_ocultas*(1-self.activaciones_ocultas)*(np.dot(delta_o,np.transpose(self.pesos2)))
			#nuevos pesos:
			pesos1_nuevos = eta*(np.dot(np.transpose(datos),delta_h[:,:-1])) + self.momentum*pesos1_nuevos
			pesos2_nuevos = eta*(np.dot(np.transpose(self.activaciones_ocultas),delta_o)) + self.momentum*pesos2_nuevos
			self.pesos1 -= pesos1_nuevos
			self.pesos2 -= pesos2_nuevos
		if grafica_curva_de_apr == True:
			plt.plot(dominio, error_mse)
			plt.title("batch")
			plt.xlabel("n")
			plt.ylabel("error")
			plt.show()
	def mlp_secuencial_entrenamiento(self,entradas, targets, eta, iteraciones, tau=100, grafica_curva_de_apr = False): 
		# de acuerdo a las gráficas este método definitivamente necesita un early stopping. Implementación que no deseo hacer, mas bien implementaré el 
		# early stopping del no secuencial, del batch. Hasta aquí llega mi trabajo en secuencial. 
		if grafica_curva_de_apr == True:
			dominio     = [i for i in range(iteraciones)]
			error_mse   = []
		datos         = np.concatenate((entradas, -np.ones((self.n_datos,1))), axis = 1)
		pesos1_nuevos = np.zeros(np.shape(self.pesos1))
		pesos2_nuevos = np.zeros(np.shape(self.pesos2))
		for n in range(iteraciones):
			for k in range(self.n_datos):
				self.pred = self.mlp_forward_seq(datos[k,:]) # obtengo la predicción forward del dato en curso 
				if self.tipo_funact == "lineal":
					res1 = self.pred-targets[k]
					#res2 = self.predicciones-targets/self.n_datos <--- la diferencia esta en que aquí divide por el numero total de datos que tenemos, aunque no es necesario ayud a reescalar los datos
					delta_o = res1
				elif self.tipo_funact == "logistica":
					delta_o = self.beta*(self.pred-targets[k])*(self.pred)*(1-self.pred) #lol te amo numpy
				elif self.tipo_funact == "softmax":
					delta_o = (self.pred-targets[k])*(self.pred*(-self.pred)+self.pred)#<--- le quité la diovision entre la cantidad total de datos. 
				else:
					print("aun no implemento esta funcion de activacion")
				#ahora delta_o es un vector de activaciones, 
				#print(self.activaciones_ocultas,np.dot(delta_o,np.transpose(self.pesos2)))
				delta_h = self.beta*self.activaciones_ocultas[:-1]*(1-self.activaciones_ocultas[:-1])*(np.dot(delta_o,np.transpose(self.pesos2))[:-1]) #checkmark
				#nuevos pesos:
				#print(delta_h)
				#print(pesos1_nuevos, np.dot(datos[k,:-1].reshape((datos[k,:-1].shape[0],1)),delta_h.reshape((1,delta_h.shape[0]))))
				pesos1_nuevos = eta*np.dot(datos[k,:].reshape((datos[k,:].shape[0],1)),delta_h.reshape((1,delta_h.shape[0]))) + self.momentum*pesos1_nuevos #not
				pesos2_nuevos = eta*(np.dot(self.activaciones_ocultas.reshape(self.n_nodoc+1,1),delta_o.reshape(1,self.n_out))) + self.momentum*pesos2_nuevos #check
				self.pesos1 -= pesos1_nuevos
				self.pesos2 -= pesos2_nuevos
			self.predicciones = self.mlp_forward(datos)
			if grafica_curva_de_apr == True:
				error_mse.append(0.5*np.sum((self.predicciones-targets)**2))
		if grafica_curva_de_apr == True:
			plt.plot(dominio, error_mse)
			plt.title("secuencial")
			plt.xlabel("n")
			plt.ylabel("error")
			plt.show()
	def mlp_forward(self,entradas):
		self.activaciones_ocultas = np.dot(entradas,self.pesos1) #esto debio de cambiarlo para el algoritmo secuencial
		self.activaciones_ocultas = 1.0/(1.0+np.exp(-self.beta*self.activaciones_ocultas))
		self.activaciones_ocultas = np.concatenate((self.activaciones_ocultas,-np.ones((np.shape(entradas)[0],1))), axis = 1)
		predicciones = np.dot(self.activaciones_ocultas, self.pesos2)
		if self.tipo_funact == "lineal":
			return(predicciones)
		elif self.tipo_funact == "logistica":
			return(1.0/(1.0+np.exp(-self.beta*predicciones)))	
		elif self.tipo_funact == "softmax":
			dens = np.sum(np.exp(predicciones),axis = 1)*np.ones((1,np.shape(predicciones)[0])) # genero la matriz de [[norm1,norm1,norm1,...],[norm2,norm2,...],[...],[...],...] que es por lo que se debe dividir la matriz de predicciones
			return(np.exp(predicciones)/dens)
		else:
			print("aun no implementa esa funcion")
	def mlp_forward_seq(self,entradas):
		self.activaciones_ocultas = np.dot(entradas,self.pesos1) #esto debio de cambiarlo para el algoritmo secuencial
		self.activaciones_ocultas = 1.0/(1.0+np.exp(-self.beta*self.activaciones_ocultas))
		self.activaciones_ocultas = np.append(self.activaciones_ocultas,-1)
		predicciones = np.dot(self.activaciones_ocultas, self.pesos2)
		if self.tipo_funact == "lineal":
			return(predicciones)
		elif self.tipo_funact == "logistica":
			return(1.0/(1.0+np.exp(-self.beta*predicciones)))	
		elif self.tipo_funact == "softmax":
			dens = np.sum(np.exp(predicciones),axis = 1)*np.ones((1,np.shape(predicciones)[0])) # genero la matriz de [[norm1,norm1,norm1,...],[norm2,norm2,...],[...],[...],...] que es por lo que se debe dividir la matriz de predicciones
			return(np.exp(predicciones)/dens)
		else:
			print("aun no implementa esa funcion")
	def prediccion(self, entrada): #asumiré que no se da la entrada con el menos uno.
		entrada = np.append(entrada,-1)
		return(self.mlp_forward_seq(entrada))

		
