'''
	This script still needs fixes. The main idea is too calculate the proportion
	of stripe in pixels and compare with the real proportion in centimeters.
	The images will be selected based on the criteria that the proportion in pixels
	is close to the real proportion.
'''

import csv
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import config as cf
import collections

logging.basicConfig(level=logging.DEBUG, filename=cf.ROOT_DIR+'/escolhidas.log')

def get_immediate_subdirectories(a_dir):
	dirlist = os.listdir(a_dir)
	dirlist.sort()
	return [name for name in dirlist
		if os.path.isdir(os.path.join(a_dir, name))]

def calc():
	os.chdir(cf.IMG_DIR)
	dirs = get_immediate_subdirectories("./")
	proporcoes_corretas = {}
	proporcoes_incorretas = {}
	erros = {}
	total_img = len(dirs)
	prop9 = 9.09090909091 # 50 / 5.5
	correct_prop = prop9
	alpha = 8.8 # min threshold
	beta = 10 #  max threshold

	for d in dirs:
		try:
			with open(d+"/reference_stripe.csv", "rt") as f:
				reader = csv.DictReader(f)
				for row in reader:
					width = float(row["w-pixels"])
					print("Largura: %.4f" % width)
					height = float(row["h-pixels"])
					print("Altura: %.4f" % height)
					proporcao = float(width / height)
					print("proporcao: %.4f" % proporcao)

					print("Usando proporcao = %.3f" % correct_prop)
					print("Usando alpha %.2f e beta %.2f" % (alpha, beta))
					erro_proporcao = abs(proporcao - correct_prop)
					print("erro: %.4f" % erro_proporcao)
					if(proporcao > alpha and proporcao <= beta):
						print("Imagem %s tem proporcao correta" % d)
						proporcoes_corretas[d] = proporcao
					else:
						proporcoes_incorretas[d] = proporcao
					erros[d] = erro_proporcao

		except IOError as err:
			continue

	logging.info('proporcoes_corretas: %s', proporcoes_corretas)
	logging.info('proporcoes_incorretas: %s', proporcoes_incorretas)
	logging.info('erros: %s', erros)
	correct_img = len(proporcoes_corretas.keys())
	p = (float(correct_img) * 100.0) / float(total_img)
	erro_medio = np.mean(erros.values())
	std_erro_medio = np.std(erros.values())
	logging.info('erro_medio: %.4f e desvio padrao: %.4f', erro_medio, std_erro_medio)
	logging.info('porcentagem de images selecionadas: %.4f %%', p)
	return (proporcoes_corretas, proporcoes_incorretas, erros)

if __name__ == '__main__':
	proporcoes_corretas, proporcoes_incorretas, erros = calc()
	err_mean = np.mean(erros.values())
	err_std = np.std(erros.values())
	logging.info("Erro medio: %.5f", err_mean)
	logging.info("Desvio padrao: %.5f", err_std)