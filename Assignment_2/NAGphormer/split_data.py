import torch
import utils_data




dataset_list = ["pubmed", "corafull", "computer", "photo", "cs", "physics"]



split_style = ["de"]
sp_list = [0.6, 0.2, 0.2]
repeat_times = 10

for dataset in dataset_list:
    print(dataset)
    for ss in split_style:
        for re_id in range(repeat_times):

            file_path = "./dataset/"+ss+"/"+ss+"_"+dataset+"_"+str(re_id)+".pt"

            adj, features, labels, train_idx, val_idx, test_idx = utils_data.load_graph(dataset, sp_list, False, False)

            data_list = [adj, features, labels, train_idx, val_idx, test_idx]

            torch.save(data_list, file_path)



