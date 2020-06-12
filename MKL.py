

# from main import experiment_setting
from general_lib import *
from similarities import get_s_metric


from MKLpy.preprocessing import kernel_normalization
from MKLpy.model_selection import train_test_split
from MKLpy.model_selection import cross_val_score
from MKLpy.algorithms import EasyMKL

from sklearn.svm import SVR, SVC
from itertools import product


from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel as hpk


def MKL():
	fname, pv, tv, org_metrics = experiment_setting()
	print (fname, pv, tv)

	list_pair_metrics =[["l1", "l2"]]

	for metrics in list_pair_metrics:
		X, y, sim_matrices = get_s_metric(
			fname=fname, tv=tv, pv=pv, 
			metrics=metrics)

		# # from similarity to kernel matrix
		KL = [np.exp(s)/0.01 for s in sim_matrices]
		KL_norm = [kernel_normalization(K) for K in KL]
		print (KL_norm, sim_matrices)

	# KLtr, KLte, Ytr, Yte = train_test_split(KL, Y, random_state=42, shuffle=True, test_size=.3)
	print (y)

	# # polynomial kernel
	# KL_norm = [hpk(X, degree=d) for d in range(1,11)]

	gamma_values = [0.001, 0.01, 0.1, 1, 10]
	
	lam_values = [0, 0.1, 0.2, 1]	
	C_values   = [0.01, 1, 100]
	for lam in lam_values:
		for gamma, C in product(gamma_values, C_values):
		    svm = SVR(kernel="rbf", C=C, gamma=gamma)
		    mkl = EasyMKL(lam=lam, learner=svm)
		    scores = cross_val_score(KL_norm, y, mkl, n_folds=3, scoring='mae')
		    print (lam, C, scores)

	# for lam, C in product(lam_values, C_values):
	#     svm = SVC(C=C)
	#     mkl = EasyMKL(lam=lam, learner=svm)
	#     # # add into MKL sources
	#     scores = cross_val_score(KL_norm, y, mkl, n_folds=3, scoring='mae') 
	#     print (lam, C, scores)


if __name__ == "__main__":
	MKL()