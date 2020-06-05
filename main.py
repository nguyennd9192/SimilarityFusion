

import pandas as pd
from sklearn.model_selection import train_test_split


from general_lib import *
from MainExample import DeepMDA
from distance_metric import get_dmetric
from NDD import SNF

def get_s_metric():
	fname = input_dir + "/Tc/TC_data_101_max.csv"
	df = pd.read_csv(fname, index_col=0)
	pv = ["Z_R", "Z_T", "C_R"]
	tv = "Tc"

	X = df[pv].values
	y = df[tv].values
	index = df.index
	num_indices = range(len(y))
	test_size = 0.3

	X = get_Xnorm(X_matrix=X)
	# X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, num_indices,
	# 				test_size=test_size, random_state=0)

	metrics = ["euclidean", "l1", "l2", "cosine", "braycurtis", "canberra", 
				"correlation", "jaccard", "hamming", "dice"]
	for metric in metrics:
		d_metric = get_dmetric(X, metric=metric)
		s_metric = 1 - d_metric 

		saveat = input_dir + "/Tc/normal_s_metric/" + metric + ".csv"
		sim_df = pd.DataFrame(s_metric, index=index, columns=index)
		makedirs(saveat)
		sim_df.to_csv(saveat)
		print (metric)

def smetric_to_snf(): 
	metrics = ["euclidean", "l1", "l2",  "correlation"] #   "cosine", "braycurtis"
	Wall = []
	for metric in metrics:
		s_metric_file = input_dir + "/Tc/normal_s_metric/" + metric + ".csv" 
		df = pd.read_csv(s_metric_file, index_col=0)
		Wall.append(df.values)

	W1 = SNF(Wall=Wall[2:], K=20, t=10 ,ALPHA=1)
	print ("W1:", W1)


	W2 = SNF(Wall=Wall[:2], K=20, t=10 ,ALPHA=1)

	print ("W2:", W2)

	print ("diff", sum(np.abs(W1 - W2)))
	print (W2.shape)




def main():
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
	main()

	# # convert representation to distance matrix
	# get_s_metric()

	# # fusion multiple similarity matrix by snf
	# smetric_to_snf()



