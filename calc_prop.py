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
	prop9 = 9.09090909091
	correct_prop = prop9
	alpha = 8.8
	beta = 10

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
	# os.chdir(cf.IMG_DIR)
	# dirs = get_immediate_subdirectories("./")
	# logging.info("dirs: %s", dirs)
	# d = {
	# 	'1':['DSCN3448AlbertPinheiroBarboza','DSCN3449AlbertPinheiroBarboza','DSCN3450AlbertPinheiroBarboza','DSCN3451AlbertPinheiroBarboza','DSCN3452AlbertPinheiroBarboza'],
	# 	'2':['DSCN3456RobertPinheiroBarboza','DSCN3457RobertPinheiroBarboza','DSCN3458RobertPinheiroBarboza','DSCN3460RobertPinheiroBarboza','DSCN3461RobertPinheiroBarboza','DSCN3462RobertPinheiroBarboza'],
	# 	'3':['DSCN3726','DSCN3727','DSCN3728'],
	# 	'4':['DSCN3737','DSCN3738','DSCN3739'],
	# 	'5':['DSCN3746','DSCN3747'],
	# 	'6':['DSCN3753','DSCN3754','DSCN3755'],
	# 	'7':['DSCN3765','DSCN3766','DSCN3767'],
	# 	'8':['DSCN3774','DSCN3775','DSCN3776'],
	# 	'9':['DSCN3783','DSCN3784','DSCN3785'],
	# 	'10':['DSCN3790','DSCN3791','DSCN3792','DSCN3793'],
	# 	'11':['DSCN3797','DSCN3798','DSCN3799','DSCN3800'],
	# 	'12':['DSCN3805','DSCN3806','DSCN3807','DSCN3808','DSCN3809','DSCN3810','DSCN3813'],
	# 	'13':['DSCN3817','DSCN3818','DSCN3819','DSCN3820','DSCN3821'],
	# 	'14':['DSCN3824','DSCN3825','DSCN3826','DSCN3827','DSCN3828'],
	# 	'15':['DSCN3831','DSCN3832','DSCN3833','DSCN3834','DSCN3835','DSCN3836','DSCN3837'],
	# 	'16':['DSCN3840','DSCN3841','DSCN3842','DSCN3843','DSCN3844'],
	# 	'17':['DSCN3847','DSCN3848','DSCN3849','DSCN3850','DSCN3851'],
	# 	'18':['DSCN3865','DSCN3866','DSCN3867','DSCN3868','DSCN3869'],
	# 	'19':['DSCN3873','DSCN3874','DSCN3875','DSCN3876','DSCN3877'],
	# 	'20':['DSCN3892','DSCN3893','DSCN3894','DSCN3895','DSCN3896'],
	# 	'21':['DSCN3900','DSCN3901','DSCN3902','DSCN3903','DSCN3904','DSCN3905'],
	# 	'22':['DSCN3910','DSCN3911','DSCN3912','DSCN3913','DSCN3914'],
	# 	'23':['DSCN3925','DSCN3926','DSCN3927','DSCN3928','DSCN3929'],
	# 	'24':['DSCN3938','DSCN3939','DSCN3940'],
	# 	'25':['DSCN3946','DSCN3947','DSCN3948'],
	# 	'26':['DSCN3953','DSCN3954','DSCN3955'],
	# 	'27':['DSCN3962','DSCN3963','DSCN3964'],
	# 	'28':['DSCN3971','DSCN3972','DSCN3973'],
	# 	'29':['DSCN3979','DSCN3980','DSCN3981','DSCN3982'],
	# 	'30':['DSCN3987','DSCN3988','DSCN3989','DSCN3990'],
	# 	'31':['DSCN3996','DSCN3997','DSCN3998','DSCN3999'],
	# 	'32':['DSCN4006','DSCN4007','DSCN4008','DSCN4009','DSCN4010','DSCN4011'],
	# 	'33':['DSCN4016','DSCN4017','DSCN4018','DSCN4019','DSCN4020','DSCN4021'],
	# 	'34':['DSCN4025','DSCN4026','DSCN4027','DSCN4028','DSCN4029'],
	# 	'35':['DSCN4086','DSCN4087','DSCN4088','DSCN4089','DSCN4090'],
	# 	'36':['DSCN4093','DSCN4094','DSCN4095','DSCN4096','DSCN4097'],
	# 	'37':['DSCN4100','DSCN4101','DSCN4102','DSCN4103','DSCN4104'],
	# 	'38':['DSCN4109','DSCN4110','DSCN4111','DSCN4112','DSCN4113'],
	# 	'39':['DSCN4117','DSCN4118','DSCN4119','DSCN4120','DSCN4121'],
	# 	'40':['DSCN4124','DSCN4125','DSCN4126','DSCN4127','DSCN4128'],
	# 	'41':['DSCN4131','DSCN4132','DSCN4133','DSCN4134','DSCN4135'],
	# 	'42':['DSCN4138','DSCN4139','DSCN4140'],
	# 	'43':['DSCN4143','DSCN4144','DSCN4145','DSCN4146','DSCN4147','DSCN4148'],
	# 	'44':['DSCN4151','DSCN4152','DSCN4153','DSCN4154','DSCN4155'],
	# 	'45':['DSCN4158','DSCN4162'],
	# 	'46':['DSCN4163','DSCN4164','DSCN4165','DSCN4166','DSCN4167'],
	# 	'47':['DSCN4173','DSCN4174','DSCN4175','DSCN4176','DSCN4177'],
	# 	'48':['DSCN4182','DSCN4183','DSCN4184','DSCN4185','DSCN4186','DSCN4187'],
	# 	'49':['DSCN4192','DSCN4193','DSCN4194','DSCN4195'],
	# 	'50':['DSCN4198','DSCN4199','DSCN4200','DSCN4201','DSCN4202'],
	# 	'51':['DSCN4208','DSCN4209','DSCN4210','DSCN4211','DSCN4212'],
	# 	'52':['DSCN4215','DSCN4216','DSCN4217','DSCN4218','DSCN4219'],
	# 	'53':['DSCN4222','DSCN4223','DSCN4224','DSCN4225','DSCN4226'],
	# 	'54':['DSCN4231','DSCN4232','DSCN4233','DSCN4234','DSCN4235','DSCN4236'],
	# 	'55':['DSCN4239','DSCN4240','DSCN4241','DSCN4242','DSCN4243'],
	# 	'56':['DSCN4246','DSCN4247','DSCN4248','DSCN4249','DSCN4250'],
	# 	'57':['DSCN4254','DSCN4255','DSCN4256','DSCN4257','DSCN4258','DSCN4259'],
	# 	'58':['DSCN4262','DSCN4263','DSCN4264','DSCN4265','DSCN4266'],
	# 	'59':['DSCN4270','DSCN4271','DSCN4272','DSCN4273','DSCN4274','DSCN4275'],
	# 	'60':['DSCN4278','DSCN4279','DSCN4280','DSCN4281','DSCN4282','DSCN4283'],
	# 	'61':['DSCN4286','DSCN4287','DSCN4288','DSCN4289','DSCN4290'],
	# 	'62':['DSCN4293','DSCN4294','DSCN4295','DSCN4296','DSCN4297'],
	# 	'63':['DSCN4300','DSCN4301','DSCN4302','DSCN4303','DSCN4304'],
	# 	'64':['DSCN4307','DSCN4308','DSCN4309','DSCN4310','DSCN4311'],
	# 	'65':['DSCN4314','DSCN4315','DSCN4316','DSCN4317','DSCN4318'],
	# 	'66':['DSCN4321','DSCN4322','DSCN4323','DSCN4324','DSCN4325'],
	# 	'67':['DSCN4328','DSCN4329','DSCN4330','DSCN4331','DSCN4332'],
	# 	'68':['DSCN4336','DSCN4337','DSCN4338','DSCN4339','DSCN4340','DSCN4341','DSCN4342'],
	# 	'69':['DSCN4345','DSCN4346','DSCN4347','DSCN4348','DSCN4349'],
	# 	'70':['DSCN4352','DSCN4353','DSCN4354','DSCN4355','DSCN4356'],
	# 	'71':['DSCN4359','DSCN4360','DSCN4361','DSCN4362','DSCN4363'],
	# 	'72':['DSCN4366','DSCN4367','DSCN4368','DSCN4369','DSCN4370'],
	# 	'73':['DSCN4373','DSCN4374','DSCN4375','DSCN4376','DSCN4377']
	# }

	proporcoes_corretas, proporcoes_incorretas, erros = calc()

	# d_chosen = {}

	# for k in d.keys():
	# 	min_val = 9999.9
	# 	chosen = ""
	# 	for i in d[k]:
	# 		if i in proporcoes_corretas and proporcoes_corretas[i] < min_val:
	# 			min_val = proporcoes_corretas[i]
	# 			chosen = i
	# 	d_chosen[chosen] = min_val
	# d_chosen = collections.OrderedDict(sorted(d_chosen.items()))
	# logging.info("Escolhidas: %s", d_chosen.keys())
	err_mean = np.mean(erros.values())
	err_std = np.std(erros.values())
	logging.info("Erro medio: %.5f", err_mean)
	logging.info("Desvio padrao: %.5f", err_std)