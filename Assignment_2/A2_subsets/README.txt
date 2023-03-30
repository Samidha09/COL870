		  ___________________________________

		   A2 DATASET REDUCTION INSTRUCTIONS
		  ___________________________________


1 Node Classification
=====================

  We are providing two scripts A2_nc_subsets.py and data_utils_nc.py.

  The interface/entry point is A2_nc_subsets.py, run it as:
  ,----
  | $ python3 A2_nc_subsets.py --dataset <dataset-name>
  `----
  <dataset-name> can be Wisconsin or Cora. The script reduces the number
  of training nodes iteratively, you need to write your code to work
  with the instance of data updated in the script. Note that the script
  doesn't save the data anywhere, it's present in memory when the script
  is running. You may adapt the script as needed. Also note, that the
  distribution of number of examples in each class is the same as in
  training set and larger sets are supersets of smaller sets.


2 Graph Similarity Learning
===========================

  For this, we provide scripts A2_ged_subsets.py and data_utils_ged.py.

  Run the interface script A2_ged_subsets.py as:
  ,----
  | $ python3 A2_ged_subsets.py --dataset <dataset-name> --ratio <ratio>
  `----
  <dataset-name> can be AIDS700nef or LINUX. <ratio> can be float in the
  closed interval [0, 1]. This file loads the train dataset and removes
  the validation graphs from the end of the train dataset, randomly
  permutes the rest and selects ratio * number_of_leftover_graphs. These
  indices are saved as a npy file. You can save graph_idxs for varying
  <ratio>'s and then load these files in your code.


3 Graph Classification
=========================

  No scripts are provided for graph classification. You can follow the
  same principles and write your own scripts.


4 Rewrite the scripts
=====================

  If needed, you may rewrite the scripts as needed, but the principles
  that each reduced subset have the same label distribution as the original
  train dataset and that large sets be supersets of smaller sets should
  be satisfied.
