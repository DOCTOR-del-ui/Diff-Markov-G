import os
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Sample')
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--addname', type=str, default=None)

    args = parser.parse_args()
    return args

def default_config(config):
    config["model"]["params"]["loss_type"] = "l1"
    config["dataloader"]["batch_size"] = 64
    config["solver"]["max_epochs"] = 10000
    config["model"]["params"]["use_markovloss"] = False
    config["model"]["params"]["use_ff"] = True
    config["model"]["params"]["use_ffloss"] = True
    config["model"]["params"]["d_model"] = 64
    config["model"]["params"]["n_heads"] = 4
    config["model"]["params"]["n_layer_dec"] = 2
    config["model"]["params"]["n_layer_enc"] = 2
    config["model"]["params"]["use_markovhead"] = False
    config["model"]["params"]["use_markovaware"] = False
    
    config["model"]["params"]["use_resid_pdrop_markovaware"] = False
    config["model"]["params"]["use_maremb_weight"] = False
    config["model"]["params"]["use_marloss_weight"] = False
    config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
    config["model"]["params"]["maremb_weight"] = 1.0
    config["model"]["params"]["marloss_weight"] = 1.0
    
    config["dataloader"]["train_dataset"]["params"]["neg_one_to_one"] = True
    
    return config

if __name__ == "__main__":
    args = parse_args()
    
    idx = args.idx
    window =args.window
    addname = args.addname
    
    with open(os.path.join("Config", "cmapss.yaml"), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    config = default_config(config)
        
    if addname == "proplus":
        config["model"]["params"]["d_model"] = 128
        config["model"]["params"]["n_heads"] = 8
        config["model"]["params"]["n_layer_dec"] = 4
        config["model"]["params"]["n_layer_enc"] = 4
        config["solver"]["max_epochs"] = 30000
        config["model"]["params"]["loss_type"] = "l2"
    elif addname == "minil1norm":
        config["solver"]["max_epochs"] = 5000
        config["dataloader"]["train_dataset"]["params"]["neg_one_to_one"] = False
        config["dataloader"]["test_dataset"]["params"]["neg_one_to_one"] = False
    elif addname == "minil1":
        config["solver"]["max_epochs"] = 5000
    elif addname == "plusl2norm":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
    elif addname == "multbh":
        config["dataloader"]["batch_size"] = 256
        config["solver"]["max_epochs"] = 2500
        config["model"]["params"]["loss_type"] = "l2"
    elif addname == "loss2":
        config["model"]["params"]["loss_type"] = "l2"
    elif addname == "norm":
        config["dataloader"]["train_dataset"]["params"]["neg_one_to_one"] = False
        config["dataloader"]["test_dataset"]["params"]["neg_one_to_one"] = False
    elif addname == "markov":
        config["solver"]["max_epochs"] = 10000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
    elif addname == "markovplus":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
    elif addname == "markovplusx":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["use_markovhead"] = True
    elif addname == "markovplusz":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovplusy":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.0
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovplusyz":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovplusy0s":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["resid_pdrop_markovaware"] = 0
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovplusyx":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
        config["model"]["params"]["use_markovaware"] = True
        config["model"]["params"]["use_markovhead"] = True
    elif addname == "markovpluswz":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["use_resid_pdrop_markovaware"] = True
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovpluswx":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["use_resid_pdrop_markovaware"] = True
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
        config["model"]["params"]["use_markovaware"] = True
        config["model"]["params"]["use_markovhead"] = True
    elif addname == "markovpluswf":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        
        config["model"]["params"]["use_resid_pdrop_markovaware"] = True
        config["model"]["params"]["use_maremb_weight"] = True
        config["model"]["params"]["use_marloss_weight"] = True
        
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovpluswf0i":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        
        config["model"]["params"]["use_resid_pdrop_markovaware"] = True
        config["model"]["params"]["use_maremb_weight"] = True
        config["model"]["params"]["use_marloss_weight"] = True
        
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["maremb_weight"] = 0
        config["model"]["params"]["marloss_weight"] = 0
        config["model"]["params"]["resid_pdrop_markovaware"] = 0
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovpluswf1i":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        
        config["model"]["params"]["use_resid_pdrop_markovaware"] = True
        config["model"]["params"]["use_maremb_weight"] = True
        config["model"]["params"]["use_marloss_weight"] = True
        
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["maremb_weight"] = 1.5
        config["model"]["params"]["marloss_weight"] = 1
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovpluswf2i":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        
        config["model"]["params"]["use_resid_pdrop_markovaware"] = True
        config["model"]["params"]["use_maremb_weight"] = True
        config["model"]["params"]["use_marloss_weight"] = True
        
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["maremb_weight"] = 1
        config["model"]["params"]["marloss_weight"] = 1
        config["model"]["params"]["resid_pdrop_markovaware"] = 1.5
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "markovplusw0s":
        config["solver"]["max_epochs"] = 15000
        config["model"]["params"]["loss_type"] = "l2"
        config["model"]["params"]["use_markovloss"] = True
        config["model"]["params"]["maremb_weight"] = 0
        config["model"]["params"]["marloss_weight"] = 0
        config["model"]["params"]["resid_pdrop_markovaware"] = 0
        config["model"]["params"]["use_markovaware"] = True
    elif addname == "nofft":
        config["model"]["params"]["use_ff"] = False
        config["model"]["params"]["use_ffloss"] = False
    elif addname == "noffloss":
        config["model"]["params"]["use_ffloss"] = False
    elif addname == "plus":
        config["solver"]["max_epochs"] = 15000
    elif addname == "simple":
        config["model"]["params"]["d_model"] = 16
        config["model"]["params"]["n_heads"] = 2
        config["model"]["params"]["n_layer_dec"] = 1
        config["model"]["params"]["n_layer_enc"] = 1
        config["solver"]["max_epochs"] = 5000
        
        
    config["model"]["params"]["seq_length"] = window
    config["dataloader"]["train_dataset"]["params"]["window"] = window
    config["dataloader"]["test_dataset"]["params"]["window"] = window

    if addname == "simple":
        config["dataloader"]["train_dataset"]["params"]["data_root"] = f"./Data/datasets/C-MAPSS-T/train_FD00{idx}.csv"
        config["dataloader"]["test_dataset"]["params"]["data_root"] = f"./Data/datasets/C-MAPSS-T/train_FD00{idx}.csv"
    else:
        config["dataloader"]["train_dataset"]["params"]["data_root"] = f"./Data/datasets/C-MAPSS/train_FD00{idx}.csv"
        config["dataloader"]["test_dataset"]["params"]["data_root"] = f"./Data/datasets/C-MAPSS/train_FD00{idx}.csv"
        
    config["solver"]["results_folder"] = f"./Checkpoints/Checkpoints_cmapsst{addname}{idx}"

    config["dataloader"]["train_dataset"]["params"]["name"] = f"cmapsst{addname}{idx}"
    config["dataloader"]["test_dataset"]["params"]["name"] = f"cmapsst{addname}{idx}"

    with open(os.path.join("Config", "cmapss.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
        