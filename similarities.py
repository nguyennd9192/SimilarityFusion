import pandas as pd
from general_lib import *
from sklearn.metrics import pairwise_distances



def get_dmetric(X, metric):
	d_matrix = pairwise_distances(X, metric=metric)
	return d_matrix

def get_s_metric(fname, tv, pv, metrics):
	# # from a given dataset, generate different similarity matrix
	df = pd.read_csv(fname, index_col=0)

	X = df[pv].values
	y = df[tv].values
	index = df.index
	num_indices = range(len(y))
	test_size = 0.3

	X = get_Xnorm(X_matrix=X)
	# X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, num_indices,
	# 				test_size=test_size, random_state=0)

	
	sim_matrices = []
	for metric in metrics:
		d_metric = get_dmetric(X, metric=metric)
		d_metric = get_Xnorm(X_matrix=d_metric)

		s_metric = 1 - d_metric 

		saveat = get_basename(fname).replace(".csv", "_") + metric + ".csv"
		sim_df = pd.DataFrame(s_metric, index=index, columns=index)
		makedirs(saveat)
		sim_df.to_csv(saveat)
		print (metric)
		sim_matrices.append(s_metric)


	return X, y, sim_matrices