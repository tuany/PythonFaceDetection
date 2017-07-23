import config as cf
import logging
import os

def getLogger(loggerForFile):
	logger = logging.getLogger(os.path.basename(loggerForFile))
	if not logger.handlers:
		logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', filename=cf.ROOT_DIR+"/output/processing.log", filemode='wb', level=logging.DEBUG)
		console = logging.StreamHandler()
		console.setLevel(logging.DEBUG)
		# set a format which is simpler for console use
		formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
		# tell the handler to use this format
		console.setFormatter(formatter)
		# add the handler to the root logger
		logger.addHandler(console)
	return logger

if __name__ == '__main__':
	print("Logging config module")