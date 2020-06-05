from sklearn.metrics import pairwise_distances



def get_dmetric(X, metric):
	d_matrix = pairwise_distances(X, metric=metric)
	return d_matrix