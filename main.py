
input_dir = "/Users/nguyennguyenduong/Dropbox/My_code/SimilarityFusion/NDD"


from MainExample import DeepMDA


def main2():
	


def main():
	# # drug_drug_matrix.csv # # to save interaction index
	index_files = ["drug_pathway_index.txt", "drug_SideEffect_index.txt", "drug_target_index.txt",
			"drug_transporter_index.txt", "drug_enzyme_index.txt", "drug_list.txt", "drug_offSideEffect_index.txt"]
	DS1_sim_files = [ 
		"DS1/chem_Jacarrd_sim.csv", "DS1/pathway_Jacarrd_sim.csv", "DS1/sideeffect_Jacarrd_sim.csv",
		"DS1/target_Jacarrd_sim.csv", "DS1/enzyme_Jacarrd_sim.csv", " DS1/transporter_Jacarrd_sim.csv",
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
	# main()

	main2()



