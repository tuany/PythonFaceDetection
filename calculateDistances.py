from scipy.spatial import distance as dist
import numpy as np

def distances(points, reference):
	distances = {}

	distances["ex_en"] = (dist.euclidean((points["x_37"], points["y_37"]), (points["x_40"], points["y_40"]))) / reference["pixelsPerMetric"]
	distances["en_ex"] = (dist.euclidean((points["x_43"], points["y_43"]), (points["x_46"], points["y_46"]))) / reference["pixelsPerMetric"]
	distances["ex_ex"] = (dist.euclidean((points["x_37"], points["y_37"]), (points["x_46"], points["y_46"]))) / reference["pixelsPerMetric"]
	distances["en_en"] = (dist.euclidean((points["x_40"], points["y_40"]), (points["x_43"], points["y_43"]))) / reference["pixelsPerMetric"]
	distances["n_Gn"]  = (dist.euclidean((points["x_9"], points["y_9"]), (points["x_28"], points["y_28"]))) / reference["pixelsPerMetric"] 
	distances["t_t"]   = (dist.euclidean((points["x_1"], points["y_1"]), (points["x_17"], points["y_17"]))) / reference["pixelsPerMetric"]
	return distances