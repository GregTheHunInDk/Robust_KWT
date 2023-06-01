from argparse import ArgumentParser
from config_parser import get_config
import os
import yaml
import matplotlib.pyplot as plt
import time

import torch
from torch import nn, optim
import wandb
from typing import Callable, Tuple 
from utils.loss import LabelSmoothingLoss
from utils.opt import get_optimizer, get_adversarial_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.trainer import train, evaluate, evaluate_sliding_window, train_single_batch, forward_single_batch, backward_single_batch
from utils.dataset import get_loader, get_noisy_loader
from utils.misc import seed_everything, count_params, get_model, calc_step, log
from utils.misc import log, save_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from collections import OrderedDict
import utils.trainer



    
    
def adv_pretrain(config, args):
    """Adverserial pretraining with noisy data.

    Args:
        config_kwt (dict): Keyword transformer config for friendly and adversarial model
        k (int): Lower K common layers that the two models share. 
    """

    ######################################
    # save hyperparameters for current run
    ######################################
    
    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    
    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    
    k = config["hparams"]["model_adv"]["k"]
    alpha = config["hparams"]["model_adv"]["alpha"]
    
    ######################################
    # data loaders
    ######################################
    
    # training and validation data is loaded from the adversarial config files
    
    # training
    # adverarial 
    print("Loading adversarial training dataset...")
    with open(config["train_list_adv_file"], "r") as f:
        train_lista = f.read().rstrip().split("\n")
    trainloadera = get_noisy_loader(train_lista, config, train=True)
    
    # friendly
    print("Loading friendly training dataset...")
    with open(config["train_list_file"], "r") as f:
        train_listf = f.read().rstrip().split("\n")
    trainloaderf = get_loader(train_listf, config, train=True)
    
    # validation
    # adversarial
    print("Loading adversarial validation dataset...")
    with open(config["val_list_adv_file"], "r") as f:
        val_lista = f.read().rstrip().split("\n")
    valloadera = get_noisy_loader(val_lista, config, train=False)
    
    print("Loading friendly validation dataset...")
    with open(config["val_list_file"], "r") as f:
        val_listf = f.read().rstrip().split("\n")
    valloaderf = get_loader(val_listf, config, train=False)
    
    
    
    
    
    
    ######################################
    # models
    ######################################
    
    # KWT adversarial model
    model_kwta = get_model(config["hparams"]["model_adv"])
    model_kwta = model_kwta.to(config["exp"]["device"])
    
    # KWT friendly model
    model_kwtf = get_model(config["hparams"]["model"])
    model_kwtf = model_kwtf.to(config["exp"]["device"])
    
    # Loading checkpoints for friendly and adversarial models
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model_kwtf.load_state_dict(ckpt["model_state_dict"])
        print("Loaded checkpoint for friendly model")
        original_dict = ckpt["model_state_dict"]
        custom_dict = OrderedDict()
        for param in original_dict:
            if not "mlp_head.1." in param:
                custom_dict[param] = original_dict[param]
        model_kwta.load_state_dict(custom_dict, strict=False)
        print("Loaded checkpoint for adversarial model")

    
    criterion_kwta = nn.CrossEntropyLoss()
    criterion_kwtf = nn.CrossEntropyLoss()

    optimizer_kwta = get_adversarial_optimizer(model_kwta, config["hparams"]["optimizer"], k, alpha)
    optimizer_kwtf = get_optimizer(model_kwtf, config["hparams"]["optimizer"])
    
    # Learning rate scheduler
    
    schedulers_kwta = {
        "warmup": None,
        "scheduler": None
    }
    
    schedulers_kwtf = {
        "warmup": None,
        "scheduler": None
    }
    
    
    ######################################
    # train models
    ######################################
    
    
    step = 0
    best_acc = 0.0
    device = config["exp"]["device"]
    log_file = os.path.join(config["exp"]["exp_dir"], "training_log.txt")
    best_avg_loss = 0.0
    n_batches = len(trainloadera)
    
    for epoch in range(config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss_kwta = 0.0
        correct_kwta = 0
        running_loss_kwtf = 0.0
        running_target_var_kwtf = 0.0
        running_prediction_var_kwtf = 0.0
        correct_kwtf = 0
        
        for (dataf, targetsf), (dataa, targetsa) in zip(trainloaderf, trainloadera):
            batch_size = dataf.size(dim=0)
            audio_length = dataf.size(dim=-1)
            
            ###################################### 
            # data2vec step - friendly step
            ######################################

            friendly_loss, friendly_outputs = forward_single_batch(model_kwtf, dataf, targetsf, device, optimizer_kwtf, criterion_kwtf)
            loss_kwtf, corr_kwtf = backward_single_batch(model_kwtf, friendly_outputs, targetsf, device, optimizer_kwtf, criterion_kwtf, friendly_loss)
            
            running_loss_kwtf += loss_kwtf
            
            # learning rate scheduler
            if schedulers_kwtf["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers_kwtf["warmup"].step()
            elif schedulers_kwtf["scheduler"] is not None:
                schedulers_kwtf["scheduler"].step()

            ###################################### 
            # kwt step - adversarial step
            ######################################

            adversarial_loss, adversarial_outputs = forward_single_batch(model_kwta, dataa, targetsa, device, optimizer_kwta, criterion_kwta)
            
            ######################################
            # backpropagation step
            ######################################
            
            # loading first K layers of freindly KWT encoder
            kwtf_partial_state_dict = load_partial_state_dict(model_kwtf.state_dict(), k)
            model_kwta.load_state_dict(kwtf_partial_state_dict, strict=False)
            
            loss_kwta, corr_kwta = backward_single_batch(model_kwta, adversarial_outputs, targetsa, device, optimizer_kwta, criterion_kwta, adversarial_loss)
            
            if not step % config["exp"]["log_freq"]:
                log_dict = {"epoch": epoch, "loss": loss_kwta, "lr": optimizer_kwta.param_groups[0]["lr"]}
            running_loss_kwta += loss_kwta
            correct_kwta += corr_kwta
            
            # logging KWT step
            if schedulers_kwta["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers_kwta["warmup"].step()
            elif schedulers_kwta["scheduler"] is not None:
                schedulers_kwta["scheduler"].step()
                
            # loading first K transformer layers of adversarial KWT
            kwta_partial_state_dict = load_partial_state_dict(model_kwta.state_dict(), k)
            model_kwtf.load_state_dict(kwta_partial_state_dict, strict=False)
            
            step += 1
            
        #################################################
        # epoch complete - log, validation and save model
        ################################################# 
        
        # data2vec log, validation and save model
        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0,
                    "avg_train_target_var": running_target_var_kwtf / n_batches,
                    "avg_train_prediction_var": running_prediction_var_kwtf / n_batches,
                    "avg_loss_per_ep": running_loss_kwta / len(trainloaderf.dataset)}
        log(log_dict, step, config)    
        
        if not epoch % config["exp"]["val_freq"]:
            val_acc, avg_val_loss = utils.trainer.evaluate(model_kwtf, criterion_kwtf,valloaderf, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            #log(log_dict, step, config_kwtf)
            
            # save best validation checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, val_acc, save_path, model_kwtf, optimizer_kwtf, log_file) 
                
        # kwt log, validation and save model
        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0, "train_acc": correct_kwta/(len(trainloadera.dataset)), "avg_loss_per_ep": running_loss_kwta/len(trainloadera)}
        log(log_dict, step, config)
        
        if not epoch % config["exp"]["val_freq"]:
            val_acc, avg_val_loss = utils.trainer.evaluate(model_kwta, criterion_kwta, valloadera, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            log(log_dict, step, config)
            
            # save best val ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config["exp"]["exp_dir"], "best.pth")
                save_model(epoch, val_acc, save_path, model_kwta, optimizer_kwta, log_file) 
    
    
    # training complete
    
    # freindly KWT final evaluation
    val_acc, avg_val_loss = evaluate(model_kwta, criterion_kwtf, valloadera, device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
    log(log_dict, step, config)

    # kwt save final checkpoint 
    save_path = os.path.join(config["exp"]["exp_dir"], "last.pth")
    save_model(epoch, val_acc, save_path, model_kwta, optimizer_kwta, log_file)
    
    
    
def load_partial_state_dict(state_dict, K):
    """_summary_

    Args:
        state_dict (_type_): state_dict to load
        K (_type_): first K layers to load
    """
    before_transformer = True
    custom_dict = OrderedDict()
    for param in state_dict:
        if param.split('.')[0] == "transformer":
            before_transformer = False
            layer_num = int(param.split('.')[2])
            if layer_num >= K:
                continue
            custom_dict[param] = state_dict[param]
        #elif before_transformer:
        #    custom_dict[param] = state_dict[param]
    #print("###################################")
    #for param in custom_dict:
    #    print(param, '\t\t', custom_dict[param].size())
    return custom_dict


def main(args):
    config_kwt = get_config(args.conf)
    seed_everything(config_kwt['hparams']['seed'])
    
    if config_kwt["exp"]["wandb"]:
        if config_kwt["exp"]["wandb_api_key"] is not None:
            with open(config_kwt["exp"]["wandb_api_key"], "r") as f:
                os.environ["WARDB_API_KEY"] = f.read()
                
        elif os.environ.get("WANDB_API_KEY") is None:
            print("Wandb API key not found.")
        
        else:
            wandb.login()
            
        with wandb.init(project=config_kwt["exp"]["proj_name"], name=config_kwt["exp"]["exp_name"], config=config_kwt["hparams"]):
            adv_pretrain(config_kwt, args)
    
    else:
        adv_pretrain(config_kwt, args)
    


if __name__ == "__main__":
    parser = ArgumentParser("Adversarial pretraining")
    parser.add_argument("--conf", type=str, required=True, help="Path to adversarial KWT config .yaml file")
    #parser.add_argument("--k", type=int, required=True, help="First K transformer layers to update")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint to load")
    args = parser.parse_args()
    main(args)