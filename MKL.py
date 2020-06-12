

# from main import experiment_setting
from general_lib import *
from similarities import get_s_metric


from MKLpy.preprocessing import kernel_normalization



def MKL():
	fname, pv, tv, org_metrics = experiment_setting()


	list_pair_metrics =[["l1", "l2"]]

	for metrics in list_pair_metrics:
		X, y, sim_matrices = get_s_metric(
			fname=fname, tv=tv, pv=pv, 
			metrics=metrics)

	KL_norm = [kernel_normalization(K) for K in sim_matrices]
	print (KL_norm)

if __name__ == "__main__":
	MKL()