from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from itertools import izip, count 
from random import randint
import numpy as np
import logger
import cv2
import imutils
import os
import random
import config as cf

log = logger.getLogger(__file__)

# XXX colocar essa funcao num script utils
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# qtd total de distancias possiveis: 2278
# 68 pontos que combinados 2 a 2 (distancia), por arranjo simples sem repeticao
# C(m,p) = m! / ((m-p)! * p!)
# C(68,2) = 68! / ((68 - 2)! * 2!) = 2278
def farkas(points, reference):
	# esta estrutura descreve as distancias baseadas no livro do Farkas, ou seja,
	# NAO possui todas as distancias possiveis.
	distances = {}
	log.info("Calculating Farkas et al distances.")
	# distancia do canto externo ao canto interno olho esquerdo.
	# As distancias em FARKAS et al. sao ex e en dos 2 lados porem coloquei indica-
	# cao de esquerda e direita
	distances["exl_enl"]    = (dist.euclidean((points["x_36"], points["y_36"]),   (points["x_39"], points["y_39"])))  /  reference["pixelsPerMetric"]
	
	# distancia do canto interno ao canto externo olho direito
	distances["enr_exr"]    = (dist.euclidean((points["x_42"], points["y_42"]),   (points["x_45"], points["y_45"])))  /  reference["pixelsPerMetric"]
	
	# distancia do canto externo ao canto externo olho esquerdo ao direito
	distances["exl_exr"]    = (dist.euclidean((points["x_36"], points["y_36"]),    (points["x_45"],  points["y_45"]))) /  reference["pixelsPerMetric"]
	
	# distancia intercantal
	distances["enl_enr"]    = (dist.euclidean((points["x_39"], points["y_39"]),    (points["x_42"],  points["y_42"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto entre os olhos ate o queixo
	distances["n_gn"]       = (dist.euclidean((points["x_8"],  points["y_8"]),     (points["x_27"],  points["y_27"]))) /  reference["pixelsPerMetric"] 
	
	# largura mandibula. FARKAS et al. pag 46
	# 'l' indica 'left' e 'r' indica 'right'
	distances["gol_gor"]    = (dist.euclidean((points["x_4"],  points["y_4"]),     (points["x_12"],  points["y_12"]))) /  reference["pixelsPerMetric"]
	
	# largura base do cranio. FARKAS et al. pag 46
	# 'l' indica 'left' e 'r' indica 'right'
	distances["zyl_zyr"]    = (dist.euclidean((points["x_2"],  points["y_2"]),     (points["x_14"],  points["y_14"]))) /  reference["pixelsPerMetric"]

	# comprimento nariz. Ponto entre o meio dos olhos e ponta do nariz
	distances["n_prn"]      = (dist.euclidean((points["x_27"], points["y_27"]),    (points["x_32"],  points["y_32"]))) /  reference["pixelsPerMetric"]

	# Distancia entre os pontos medianos da palpebra superior e inferior do olho esquerdo. 
	# Calculo diferente devido a localizacao dos pontos de acordo com saida
	# e definicao em FARKAS et al
	distances["psl_pil_1"]  = (dist.euclidean((points["x_37"], points["y_37"]),    (points["x_41"], points["y_41"]))) /  reference["pixelsPerMetric"]
	distances["psl_pil_2"]  = (dist.euclidean((points["x_38"], points["y_38"]),    (points["x_40"], points["y_40"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano palpebra superior ao canto externo olho esquerdo
	distances["psl_exl_1"]  = (dist.euclidean((points["x_37"], points["y_37"]),     (points["x_36"],  points["y_36"]))) /  reference["pixelsPerMetric"]
	distances["psl_exl_2"]  = (dist.euclidean((points["x_38"], points["y_38"]),     (points["x_36"],  points["y_36"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["psl_enl_1"]  = (dist.euclidean((points["x_37"], points["y_37"]),     (points["x_39"],  points["y_39"]))) /  reference["pixelsPerMetric"]
	distances["psl_enl_2"]  = (dist.euclidean((points["x_38"], points["y_38"]),     (points["x_39"],  points["y_40"]))) /  reference["pixelsPerMetric"]

	# distancia ponto mediano palpebra inferior ao canto externo olho esquerdo
	distances["pil_exl_1"]  = (dist.euclidean((points["x_41"], points["y_41"]),     (points["x_36"],  points["y_36"]))) /  reference["pixelsPerMetric"]
	distances["pil_exl_2"]  = (dist.euclidean((points["x_40"], points["y_40"]),     (points["x_36"],  points["y_36"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["pil_enl_1"]  = (dist.euclidean((points["x_41"], points["y_41"]),     (points["x_39"],  points["y_39"]))) /  reference["pixelsPerMetric"]
	distances["pil_enl_2"]  = (dist.euclidean((points["x_40"], points["y_40"]),     (points["x_39"],  points["y_39"]))) /  reference["pixelsPerMetric"]

	# Distancia entre os pontos medianos da palpebra superior e inferior do olho direito. 
	distances["psr_pir_1"]  = (dist.euclidean((points["x_43"], points["y_43"]),    (points["x_47"], points["y_47"]))) /  reference["pixelsPerMetric"]
	distances["psr_pir_2"]  = (dist.euclidean((points["x_44"], points["y_44"]),    (points["x_46"], points["y_46"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano palpebra superior ao canto externo olho esquerdo
	distances["psr_exr_1"]  = (dist.euclidean((points["x_43"], points["y_43"]),     (points["x_45"],  points["y_45"]))) /  reference["pixelsPerMetric"]
	distances["psr_exr_2"]  = (dist.euclidean((points["x_44"], points["y_44"]),     (points["x_45"],  points["y_45"]))) /  reference["pixelsPerMetric"]

	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["psr_enr_1"]  = (dist.euclidean((points["x_43"], points["y_43"]),     (points["x_42"],  points["y_42"]))) /  reference["pixelsPerMetric"]
	distances["psr_enr_2"]  = (dist.euclidean((points["x_44"], points["y_44"]),     (points["x_42"],  points["y_42"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano palpebra inferior ao canto externo olho esquerdo
	distances["pir_exr_1"]  = (dist.euclidean((points["x_47"], points["y_47"]),     (points["x_45"],  points["y_45"]))) /  reference["pixelsPerMetric"]
	distances["pir_exr_2"]  = (dist.euclidean((points["x_46"], points["y_46"]),     (points["x_45"],  points["y_45"]))) /  reference["pixelsPerMetric"]
	
	# distancia ponto mediano pupila ao canto interno olho esquerdo
	distances["pir_enr_1"]  = (dist.euclidean((points["x_47"], points["y_47"]),     (points["x_42"],  points["y_42"]))) /  reference["pixelsPerMetric"]
	distances["pir_enr_2"]  = (dist.euclidean((points["x_46"], points["y_46"]),     (points["x_42"],  points["y_42"]))) /  reference["pixelsPerMetric"]

	# distancia do espaco entre nariz e labio superior
	distances["sn_ls"]      = (dist.euclidean((points["x_33"], points["y_33"]),    (points["x_51"],  points["y_51"]))) /  reference["pixelsPerMetric"]

	# comprimento da parte inferior do rosto, nariz ate o queixo
	distances["sn_gn"]      = (dist.euclidean((points["x_33"], points["y_33"]),    (points["x_8"],  points["y_8"]))) /  reference["pixelsPerMetric"]

	# largura de ponta a ponta da boca
	distances["chl_chr"]    = (dist.euclidean((points["x_48"], points["y_48"]),    (points["x_54"],  points["y_54"]))) /  reference["pixelsPerMetric"]

	# distancia do ponto entre o meio dos olhos e ponto medio da boca
	distances["n_sto"]      = (dist.euclidean((points["x_27"], points["y_27"]),    (points["x_62"],  points["y_62"]))) /  reference["pixelsPerMetric"]

	# parte inferior do rosto: do ponto medio da boca ate o queixo
	distances["gn_sto"]     = (dist.euclidean((points["x_8"],  points["y_8"]),     (points["x_62"], points["y_62"]))) /  reference["pixelsPerMetric"]

	# parte inferior do rosto: comprimento do queixo
	distances["si_gn"]      = (dist.euclidean((points["x_57"],  points["y_57"]),     (points["x_8"], points["y_8"]))) /  reference["pixelsPerMetric"]

	# altura do perfil inferior FARKAS et al. p 47
	distances["prn_gn"]     = (dist.euclidean((points["x_30"],  points["y_30"]),     (points["x_8"], points["y_8"]))) /  reference["pixelsPerMetric"]

	# Metade inferior da altura craniofacial FARKAS et al p 47
	distances["enl_gn"]     = (dist.euclidean((points["x_38"],  points["y_38"]),     (points["x_8"], points["y_8"]))) /  reference["pixelsPerMetric"]	

	# Metade inferior da altura craniofacial FARKAS et al p 47
	distances["enr_gn"]     = (dist.euclidean((points["x_42"],  points["y_42"]),     (points["x_8"], points["y_8"]))) /  reference["pixelsPerMetric"]	

	# distancia da glabela subnasal FARKAS et al p 47. Deveria ser 'g_sn' porem nao temos saida do ponto 'g'
	distances["n_sn"]     	= (dist.euclidean((points["x_27"],  points["y_27"]),     (points["x_33"], points["y_33"]))) /  reference["pixelsPerMetric"]		

	# distancia do canto interior esquerdo para o nariz
	distances["enl_se"]     = (dist.euclidean((points["x_39"],  points["y_39"]),     (points["x_28"], points["y_28"]))) /  reference["pixelsPerMetric"]		

	# distancia do canto interior esquerdo para o nariz
	distances["enr_se"]     = (dist.euclidean((points["x_42"],  points["y_42"]),     (points["x_28"], points["y_28"]))) /  reference["pixelsPerMetric"]		

	log.info("{0} distances calculated".format(len(distances)))
	return distances

def all(points, reference):
	distances =  {}

	for (i, j) in izip(range(0, 68), range(0, 68)):
		x1 = "x_" + str(i)
		y1 = "y_" + str(j)
		for (k, l) in izip(range(0, 68), range(0, 68)):
			key = ""

			if i == k:
				continue

			if i < k:
				key = "p" + str(i) + "_q" + str(k)
			else:
				"p" + str(k) + "_q" + str(i)

			if key != "" and key in distances:
				continue

			x2 = "x_" + str(k)
			y2 = "y_" + str(l)
			distances[key] = (dist.euclidean((points[x1], points[y1]), (points[x2], points[y2]))) / reference["pixelsPerMetric"]
	
	log.info("{0} distances calculated".format(len(distances)))
	return distances

def few(final_image_path, output_name, points, reference):
	A = "dist_intercantal_ext"
	B = "dist_intercantal_int"
	C = "dist_interpupilar"
	D = "dist_nariz_labio_sup"
	E = "dist_fenda_palpebral_esq1"
	F = "dist_fenda_palpebral_esq2"
	G = "dist_fenda_palpebral_dir1"
	H = "dist_fenda_palpebral_dir2"
	distances = {}
	distances[A] = (dist.euclidean((points["x_36"], points["y_36"]), (points["x_45"], points["y_45"]))) / reference["pixelsPerMetric"]
	distances[B] = (dist.euclidean((points["x_39"], points["y_39"]), (points["x_42"], points["y_42"]))) / reference["pixelsPerMetric"]

	m = (points["x_37"], points["y_37"])
	n = (points["x_38"], points["y_38"])
	(mnX, mnY) = midpoint(m, n)
	p = (points["x_43"], points["y_43"])
	q = (points["x_44"], points["y_44"])
	(pqX, pqY) = midpoint(p, q)
	distances[C] = (dist.euclidean((mnX, mnY), (pqX, pqY))) / reference["pixelsPerMetric"]
	distances[D] = (dist.euclidean((points["x_33"], points["y_33"]), (points["x_51"], points["y_51"]))) / reference["pixelsPerMetric"]
	distances[E] = (dist.euclidean((points["x_37"], points["y_37"]), (points["x_41"], points["y_41"]))) / reference["pixelsPerMetric"]
	distances[F] = (dist.euclidean((points["x_38"], points["y_38"]), (points["x_40"], points["y_40"]))) / reference["pixelsPerMetric"]
	distances[G] = (dist.euclidean((points["x_43"], points["y_43"]), (points["x_47"], points["y_47"]))) / reference["pixelsPerMetric"]
	distances[H] = (dist.euclidean((points["x_44"], points["y_44"]), (points["x_46"], points["y_46"]))) / reference["pixelsPerMetric"]

	refCoords = np.vstack([(points["x_36"], points["y_36"]), (points["x_39"], points["y_39"]), (mnX, mnY), (points["x_33"], points["y_33"]), (points["x_37"], points["y_37"]), (points["x_38"], points["y_38"]), (points["x_43"], points["y_43"]), (points["x_44"], points["y_44"])])
	objCoords = np.vstack([(points["x_45"], points["y_45"]), (points["x_42"], points["y_42"]), (pqX, pqY), (points["x_51"], points["y_51"]), (points["x_41"], points["y_41"]), (points["x_40"], points["y_40"]), (points["x_47"], points["y_47"]), (points["x_46"], points["y_46"])])
	plot_all_points(final_image_path, "few" + output_name, points, reference["pixelsPerMetric"], refCoords, objCoords)
	log.info("{0} distances calculated".format(len(distances)))
	return distances

def plot_all_points(final_image_path, output_name, points, pixelsPerMetric,refCoords, objCoords):
	names = []
	# refCoords = np.vstack([])

	for key, value in points.iteritems():
		names.append(key)
	# plot
	log.info("Image path {}".format(final_image_path))
	image = cv2.imread(final_image_path)
	color = (0, 0, 255) # vermelho
	if image is not None :
		orig = image.copy()
		# loop over the original points
		for ((u, v), (w, z), nam) in zip(refCoords, objCoords, names):
			# draw circles corresponding to the current points and
			# connect them with a line
			color = randomColor()
			cv2.circle(orig, (int(u), int(v)), 4, color, -1)
			cv2.circle(orig, (int(w), int(z)), 4, color, -1)
			cv2.line(orig, (int(u), int(v)), (int(w), int(z)),
			color, 1)
			D = dist.euclidean((u, v), (w, z)) / pixelsPerMetric
			(mX, mY) = midpoint((u, v), (w, z))
			cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

			# show the output image
		cv2.imwrite(cf.OUTPUT_DIR + "/" + output_name + ".jpg", orig)

def randomColor():
	return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))