from scipy.spatial import distance as dist
import numpy as np

# XXX colocar essa funcao num script utils
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# qtd total de distancias possiveis: 2278
# 68 pontos que combinados 2 a 2 (distancia), por arranjo simples sem repeticao
# C(m,p) = m! / ((m-p)! * p!)
# C(68,2) = 68! / ((68 - 2)! * 2!) = 2278
def distancesFarkas(points, reference):
	# esta estrutura descreve as distancias baseadas no livro do Farkas, ou seja,
	# NAO possui todas as distancias possiveis.
	distances = {}

	# distancia do canto externo ao canto interno olho esquerdo.
	# As distancias em FARKAS et al. sao ex e en dos 2 lados porem coloquei indica-
	# cao de esquerda e direita
	distances["exl_enl"]    = (dist.euclidean((points["x_37"], points["y_37"]),   (points["x_40"], points["y_40"])))  /  reference["pixelsPerMetric"]
	
	# distancia do canto interno ao canto externo olho direito
	distances["enr_exr"]    = (dist.euclidean((points["x_43"], points["y_43"]),   (points["x_46"], points["y_46"])))  /  reference["pixelsPerMetric"]
	
	# distancia do canto externo ao canto externo olho esquerdo ao direito
	distances["exl_exr"]    = (dist.euclidean((points["x_37"], points["y_37"]),    (points["x_46"],  points["y_46"]))) /  reference["pixelsPerMetric"]
	
	# distancia intercantal
	distances["enl_enr"]    = (dist.euclidean((points["x_40"], points["y_40"]),    (points["x_43"],  points["y_43"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto entre os olhos ate o queixo
	distances["n_gn"]       = (dist.euclidean((points["x_9"],  points["y_9"]),     (points["x_28"],  points["y_28"]))) /  reference["pixelsPerMetric"] 
	
	# largura mandibula. FARKAS et al. pag 46
	# 'l' indica 'left' e 'r' indica 'right'
	distances["gol_gor"]    = (dist.euclidean((points["x_5"],  points["y_5"]),     (points["x_13"],  points["y_13"]))) /  reference["pixelsPerMetric"]
	
	# largura base do cranio. FARKAS et al. pag 46
	# 'l' indica 'left' e 'r' indica 'right'
	distances["zyl_zyr"]    = (dist.euclidean((points["x_3"],  points["y_3"]),     (points["x_15"],  points["y_15"]))) /  reference["pixelsPerMetric"]

	# comprimento nariz. Ponto entre o meio dos olhos e ponta do nariz
	distances["n_prn"]      = (dist.euclidean((points["x_28"], points["y_28"]),    (points["x_31"],  points["y_31"]))) /  reference["pixelsPerMetric"]

	# Distancia entre os pontos medianos da palpebra superior e inferior do olho esquerdo. 
	# Calculo diferente devido a localizacao dos pontos de acordo com saida
	# e definicao em FARKAS et al
	tl = (points["x_38"], points["y_38"])
	tr = (points["x_39"], points["y_39"])
	(tltrX, tltrY) = midpoint(tl, tr)
	bl = (points["x_42"], points["y_42"])
	br = (points["x_41"], points["y_41"])
	(blbrX, blbrY) = midpoint(bl, br)
	distances["psl_pil"]      = (dist.euclidean((tltrX, tltrY),    (blbrX, blbrY))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano palpebra superior ao canto externo olho esquerdo
	distances["psl_exl"]    = (dist.euclidean((tltrX, tltrY),     (points["x_37"],  points["y_37"]))) /  reference["pixelsPerMetric"]
	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["psl_enl"]    = (dist.euclidean((tltrX, tltrY),     (points["x_40"],  points["y_40"]))) /  reference["pixelsPerMetric"]

	# distancia ponto mediano palpebra inferior ao canto externo olho esquerdo
	distances["pil_exl"]    = (dist.euclidean((blbrX, blbrY),     (points["x_37"],  points["y_37"]))) /  reference["pixelsPerMetric"]
	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["pil_enl"]    = (dist.euclidean((blbrX, blbrY),     (points["x_40"],  points["y_40"]))) /  reference["pixelsPerMetric"]

	# Distancia entre os pontos medianos da palpebra superior e inferior do olho direito. 
	tl = (points["x_44"], points["y_44"])
	tr = (points["x_45"], points["y_45"])
	(tltrX, tltrY) = midpoint(tl, tr)
	bl = (points["x_48"], points["y_48"])
	br = (points["x_47"], points["y_47"])
	(blbrX, blbrY) = midpoint(bl, br)

	distances["psr_pir"]      = (dist.euclidean((tltrX, tltrY),    (blbrX, blbrY))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano palpebra superior ao canto externo olho esquerdo
	distances["psr_exr"]    = (dist.euclidean((tltrX, tltrY),     (points["x_46"],  points["y_46"]))) /  reference["pixelsPerMetric"]
	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["psr_enr"]    = (dist.euclidean((tltrX, tltrY),     (points["x_43"],  points["y_43"]))) /  reference["pixelsPerMetric"]

	# distancia ponto mediano palpebra inferior ao canto externo olho esquerdo
	distances["pir_exr"]    = (dist.euclidean((blbrX, blbrY),     (points["x_46"],  points["y_46"]))) /  reference["pixelsPerMetric"]
	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["pir_enr"]    = (dist.euclidean((blbrX, blbrY),     (points["x_43"],  points["y_43"]))) /  reference["pixelsPerMetric"]

	# distancia do espaco entre nariz e labio superior
	distances["sn_ls"]      = (dist.euclidean((points["x_34"], points["y_34"]),    (points["x_52"],  points["y_52"]))) /  reference["pixelsPerMetric"]

	# comprimento da parte inferior do rosto, nariz ate o queixo
	distances["sn_gn"]      = (dist.euclidean((points["x_34"], points["y_34"]),    (points["x_9"],  points["y_9"]))) /  reference["pixelsPerMetric"]

	# largura de ponta a ponta da boca
	distances["chl_chr"]    = (dist.euclidean((points["x_49"], points["y_49"]),    (points["x_55"],  points["y_55"]))) /  reference["pixelsPerMetric"]

	# distancia do ponto entre o meio dos olhos e ponto medio da boca
	distances["n_sto"]      = (dist.euclidean((points["x_28"], points["y_28"]),    (points["x_63"],  points["y_63"]))) /  reference["pixelsPerMetric"]

	# parte inferior do rosto: do ponto medio da boca ate o queixo
	distances["gn_sto"]     = (dist.euclidean((points["x_9"],  points["y_9"]),     (points["x_63"], points["y_63"]))) /  reference["pixelsPerMetric"]

	# parte inferior do rosto: comprimento do queixo
	distances["si_gn"]      = (dist.euclidean((points["x_58"],  points["y_58"]),     (points["x_9"], points["y_9"]))) /  reference["pixelsPerMetric"]

	# altura do perfil inferior FARKAS et al. p 47
	distances["prn_gn"]     = (dist.euclidean((points["x_31"],  points["y_31"]),     (points["x_9"], points["y_9"]))) /  reference["pixelsPerMetric"]

	# altura do perfil inferior FARKAS et al. p 47
	distances["prn_gn"]     = (dist.euclidean((points["x_31"],  points["y_31"]),     (points["x_9"], points["y_9"]))) /  reference["pixelsPerMetric"]	

	# Metade inferior da altura craniofacial FARKAS et al p 47
	distances["enl_gn"]     = (dist.euclidean((points["x_40"],  points["y_40"]),     (points["x_9"], points["y_9"]))) /  reference["pixelsPerMetric"]	

	# Metade inferior da altura craniofacial FARKAS et al p 47
	distances["enr_gn"]     = (dist.euclidean((points["x_43"],  points["y_43"]),     (points["x_9"], points["y_9"]))) /  reference["pixelsPerMetric"]	

	# distancia da glabela subnasal FARKAS et al p 47. Deveria ser 'g_sn' porem nao temos saida do ponto 'g'
	distances["n_sn"]     	= (dist.euclidean((points["x_28"],  points["y_28"]),     (points["x_34"], points["y_34"]))) /  reference["pixelsPerMetric"]		

	# distancia do canto interior esquerdo para o nariz
	distances["enl_se"]     = (dist.euclidean((points["x_40"],  points["y_40"]),     (points["x_29"], points["y_29"]))) /  reference["pixelsPerMetric"]		

	# distancia do canto interior esquerdo para o nariz
	distances["enr_se"]     = (dist.euclidean((points["x_43"],  points["y_43"]),     (points["x_29"], points["y_29"]))) /  reference["pixelsPerMetric"]		

	return distances