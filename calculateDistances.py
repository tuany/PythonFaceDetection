from scipy.spatial import distance as dist
import numpy as np

def distances(points, reference):
	# qtd total de caracteristicas possiveis: 2278
	# 68 pontos que combinados 2 a 2 (distancia), por arranjo simples sem repeticao
	# C(m,p) = m! / ((m-p)! * p!)
	# C(68,2) = 68! / ((68 - 2)! * 2!) = 2278
	distances = {}

	# esta estrutura descreve as distancias baseadas no livro do Farkas, ou seja,
	# NAO possui todas as distancias possiveis.

	# distancia do canto externo ao canto interno olho esquerdo.
	# As distancias em FARKAS et al. sao ex e en dos 2 lados porem coloquei indica-
	# cao de esquerda e direita
	distances["exl_enl"]    = (dist.euclidean((points["x_37"], points["y_37"]), (points["x_40"], points["y_40"]))) / reference["pixelsPerMetric"]
	
	# distancia do canto interno ao canto externo olho direito
	distances["enr_exr"]    = (dist.euclidean((points["x_43"], points["y_43"]), (points["x_46"], points["y_46"]))) / reference["pixelsPerMetric"]
	
	# distancia do canto externo ao canto externo olho esquerdo ao direito
	distances["exl_exr"]    = (dist.euclidean((points["x_37"],  points["y_37"]),  (points["x_46"],  points["y_46"])))  /  reference["pixelsPerMetric"]
	
	# distancia intercantal
	distances["enl_enr"]    = (dist.euclidean((points["x_40"],  points["y_40"]),  (points["x_43"],  points["y_43"])))  /  reference["pixelsPerMetric"]
	# distancia ponto entre os olhos ate o queixo
	distances["n_Gn"]     = (dist.euclidean((points["x_9"],   points["y_9"]),   (points["x_28"],  points["y_28"])))  /  reference["pixelsPerMetric"] 
	# largura base do cranio. 
	# 'l' indica 'left' e 'r' indica 'right'
	distances["tl_tr"]      = (dist.euclidean((points["x_3"],   points["y_3"]),   (points["x_15"],  points["y_15"])))  /  reference["pixelsPerMetric"]
	
	return distances