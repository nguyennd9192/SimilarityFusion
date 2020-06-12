

import pandas as pd
import itertools
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dropout, Activation, Flatten
from keras.models import Sequential
from keras.layers import Dense


from general_lib import *
from MainExample import DeepMDA
from distance_metric import get_dmetric
from NDD import SNF, FullyConnected
from kr_parameter_search import CV_predict_score
from similarities import get_s_metric




def smetric_to_snf(): 
	# # load from existed similarity matrices
	# metrics = ["euclidean", "l1", "l2",  "correlation"] #   "cosine", "braycurtis"
	# Wall = []
	# for metric in metrics:
	# 	s_metric_file = input_dir + "/Tc/normal_s_metric/" + metric + ".csv" 
	# 	df = pd.read_csv(s_metric_file, index_col=0)
	# 	Wall.append(df.values)
	fname, pv, tv, org_metrics = experiment_setting()


	# list_pair_metrics = itertools.combinations(org_metrics, 3)
	list_pair_metrics =[["l1", "l2"]]

	for metrics in list_pair_metrics:

		X, y, sim_matrices = get_s_metric(
			fname=fname, tv=tv, pv=pv, 
			metrics=metrics)

		W = SNF(Wall=sim_matrices, K=10, t=10, ALPHA=1)
		print ("W:", W)


		# W2 = SNF(Wall=Wall[:2], K=20, t=10, ALPHA=1)

		# print ("W2:", W2)

		# print ("diff", sum(np.abs(W1 - W2)))

		W = np.array(sim_matrices[0])
		# W = W[:, 0:20]
		print (W.shape, y.shape)

		n_train = int(sim_matrices[0].shape[0])


		model = FullyConnected(input_dim=W.shape[1])

		# input_dim=W.shape[1]
		# estimator = KerasRegressor(build_fn=model.build, 
		# 		epochs=2000, batch_size=2, verbose=0)
		# kfold = KFold(n_splits=10)
		# results = cross_val_score(estimator, W, y, cv=kfold)

		# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
		# print("results", results)


		# # modify since using similarity matrix as input, we cannot have distance to test point
		r2, r2_std, mae, mae_std = CV_predict_score(model=model, X=W, y=y, 
				n_folds=10, n_times=3, score_type='r2')
		print ("r2, r2_std, mae, mae_std:", r2, r2_std, mae, mae_std)
		result = [dict({"pv":pv, "metrics": metrics, 
			"r2":r2, "r2_std":r2_std, 
			"mae":mae, "mae_std":mae_std})]
		# result_df = pd.DataFrame(result)
		# saveat = result_dir + "/Tc/{0}.csv".format("|".join(metrics))
		# makedirs(file=saveat)
		# print (saveat)
		# result_df.to_csv(saveat)



def test_NDD():
	# # drug_drug_matrix.csv # # to save interaction index
	index_files = ["drug_pathway_index.txt", "drug_SideEffect_index.txt", "drug_target_index.txt",
			"drug_transporter_index.txt", "drug_enzyme_index.txt", "drug_list.txt", "drug_offSideEffect_index.txt"]
	DS1_sim_files = [ 
		# "DS1/chem_Jacarrd_sim.csv", "DS1/pathway_Jacarrd_sim.csv", "DS1/sideeffect_Jacarrd_sim.csv",
		# "DS1/target_Jacarrd_sim.csv", "DS1/enzyme_Jacarrd_sim.csv", 
		"DS1/transporter_Jacarrd_sim.csv",
		"DS1/indication_Jacarrd_sim.csv", "DS1/offsideeffect_Jacarrd_sim.csv"
		]
	DS2_sim_files = ["DS2/simMatrix.csv", "DS2/ddiMatrix.csv"]
	DS3_sim_files = ["DS3/SideEffectSimilarityMat.csv", "DS3/seqSimilarityMat.csv", 
		"DS3/NCRDInteractionMat.csv", "DS3/ligandSimilarityMat.csv", "DS3/GOSimilarityMat.csv", 
		"DS3/distSimilarityMat.csv", "DS3/CRDInteractionMat.csv", "DS3/chemicalSimilarityMat.csv", 
		"DS3/ATCSimilarityMat.csv"
		]

	sim_files = DS1_sim_files + DS2_sim_files + DS3_sim_files

	for sf in sim_files:
		sim_file = "{0}/{1}".format(input_dir, sf)
		print (sim_file)
		DeepMDA(sim_file=sim_file)	



if __name__ == "__main__":
	# # run test drug-drug interaction
	# test_NDD()

	# # convert representation to distance matrix
	# get_s_metric()

	# # fusion multiple similarity matrix by snf
	smetric_to_snf()



