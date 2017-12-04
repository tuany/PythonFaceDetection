import csv
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import config as cf

logging.basicConfig(level=logging.DEBUG, filename=cf.ROOT_DIR+'/calculo-proporcoes2.log')

def get_immediate_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir)
		if os.path.isdir(os.path.join(a_dir, name))]

def calc():
	os.chdir(cf.IMG_DIR)
	dirs = get_immediate_subdirectories("./")
	proporcoes_corretas = {}
	proporcoes_incorretas = {}
	erros = {}
	total_img = len(dirs)

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
					erro_proporcao = abs(proporcao - 9.09090909091)
					print("erro: %.4f" % erro_proporcao)
					if(proporcao > 9 and proporcao <= 10):
						print("Imagem %s tem proporcao correta" % d)
						proporcoes_corretas[d] = proporcao
						erros[d] = erro_proporcao
					else:
						proporcoes_incorretas[d] = proporcao

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

if __name__ == '__main__':
	calc()