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
from utils.trainer import train, evaluate, evaluate_sliding_window, train_single_batch
from utils.dataset import get_loader, get_noisy_loader
from utils.misc import seed_everything, count_params, get_model, calc_step, log
#from data2vec.data2vec_utils.trainer import train_single_batch
from utils.misc import log, save_model
from torch.utils.data import DataLoader
from data2vec.masking import AudioMaskingGenerator
from models.Data2Vec import Data2Vec
from tqdm import tqdm
import copy
from collections import OrderedDict
import data2vec.data2vec_utils.trainer
import utils.trainer



    
    
def adv_pretrain(config_kwt, config_d2v, k, alpha):
    """Adverserial pretraining with noisy data.

    Args:
        config_kwt (dict): Keyword transformer config for noise type prediction
        config_d2v (dict): Data2vec config for masked prediction (regression)
        k (int): Lower K common layers that the two models share. 
    """

    ######################################
    # save hyperparameters for current run
    ######################################
    
    config_kwt["exp"]["save_dir"] = os.path.join(config_kwt["exp"]["exp_dir"], config_kwt["exp"]["exp_name"])
    os.makedirs(config_kwt["exp"]["save_dir"], exist_ok=True)
    
    config_d2v["exp"]["save_dir"] = os.path.join(config_d2v["exp"]["exp_name"])
    os.makedirs(config_d2v["exp"]["save_dir"], exist_ok=True)
    
    config_str = yaml.dump(config_kwt)
    print("Using settings:\n", config_str)

    with open(os.path.join(config_kwt["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    
    
    ######################################
    # data loaders
    ######################################
    
    # training
    # adversarial
    print("Loading adversarial training dataset...")
    with open(config_kwt["train_list_file"], "r") as f:
        train_lista = f.read().rstrip().split("\n")
    trainloadera = get_noisy_loader(train_lista, config_kwt, train=True)
    
    # friendly
    print("Loading friendly training dataset...")
    with open(config_d2v["train_list_file"], "r") as f:
        train_listf = f.read().rstrip().split("\n")
    trainloaderf = get_loader(train_listf, config_d2v, train=True)
    
    # validation
    # adversarial
    print("Loading adversarial validation dataset...")
    with open(config_kwt["val_list_file"], "r") as f:
        val_lista = f.read().rstrip().split("\n")
    valloadera = get_noisy_loader(val_lista, config_kwt, train=False)
    
    #friendly
    print("Loading friendly validation dataset...")
    with open(config_d2v["val_list_file"], "r") as f:
        val_listf = f.read().rstrip().split("\n")
    valloaderf = get_loader(val_listf, config_d2v, train=False)
    
    mask_generator = AudioMaskingGenerator(mask_prob=config_d2v["hparams"]["model"]["mask_prob"],
                                           mask_length=config_d2v["hparams"]["model"]["mask_length"],
                                           attention_mask=None,
                                           min_masks=config_d2v["hparams"]["model"]["min_masks"])
    
    ######################################
    # models
    ######################################
    
    # KWT model
    model_kwt = get_model(config_kwt["hparams"]["model"])
    model_kwt = model_kwt.to(config_kwt["hparams"]["device"])
    
    model_kwt_copy = copy.deepcopy(model_kwt)
    
    # data2vec model
    model_d2v = Data2Vec(encoder=model_kwt_copy,
                        modality=config_d2v["modality"],
                        model_embed_dim=config_d2v["hparams"]["model"]["dim"],
                        ema_decay=config_d2v["hparams"]["model"]["ema_decay"],
                        ema_end_decay=config_d2v["hparams"]["model"]["ema_end_decay"],
                        ema_anneal_end_step=config_d2v["hparams"]["model"]["ema_anneal_end_step"],
                        average_top_k_layers=config_d2v["hparams"]["model"]["average_top_k_layers"],
                        normalize_targets=config_d2v["hparams"]["model"]["normalize_targets"])
    model_d2v= model_d2v.to(config_d2v["hparams"]["device"])
    
    criterion_kwt = nn.CrossEntropyLoss()
    criterion_d2v = nn.MSELoss(reduction="none")
    
    parameters_kwt = model_kwt.parameters()
    parameters_d2v = model_d2v.parameters()
    
    
    # optimizer for KWT
    optimizer_kwt = get_adversarial_optimizer(model_kwt, config_kwt["hparams"]["optimizer"], k, alpha)
    
    # optimizer for data2vec
    optimizer_d2v = optim.Adam(parameters_d2v, lr=config_d2v["hparams"]["optimizer"]["opt_kwargs"]["lr"],
                           betas=config_d2v["hparams"]["optimizer"]["opt_kwargs"]["betas"],
                           eps=config_d2v["hparams"]["optimizer"]["opt_kwargs"]["eps"],
                           weight_decay=config_d2v["hparams"]["optimizer"]["opt_kwargs"]["weight_decay"])
    
    
    #for group in optimizer_kwt.param_groups:
    #    print(group['lr'])
        
    #return
    
    # Learning rate scheduler for data2vec
    epochs = config_d2v["hparams"]["n_epochs"]
    steps_per_epoch = len(trainloaderf)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer_d2v,
        max_lr=config_d2v["hparams"]["optimizer"]["opt_kwargs"]["lr"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        anneal_strategy="cos")
    schedulers_d2v = {"scheduler": lr_scheduler,
                  "warmup": 0}
    
    # Learning rate scheduler for KWT
    schedulers_kwt = {
        "warmup": None,
        "scheduler": None
    }
    
    # Setting up the learning rate scheduler for data2vec and KWT
    if config_d2v["hparams"]["scheduler"]["n_warmup"]:
        schedulers_d2v["warmup"] = WarmUpLR(optimizer_kwt, total_iters=len(trainloaderf) * config_d2v["hparams"]["scheduler"]["n_warmup"])
        
    if config_d2v["hparams"]["scheduler"]["scheduler_type"] is not None: 
        total_iters = len(trainloaderf) * max(1, (config_d2v["hparams"]["scheduler"]["max_epochs"] - config_d2v["hparams"]["scheduler"]["n_warmup"]))
        schedulers_d2v["scheduler"] = get_scheduler(optimizer_kwt, config_d2v["hparams"]["scheduler"]["scheduler_type"], total_iters)
        
    if config_kwt["hparams"]["scheduler"]["n_warmup"]:
        schedulers_kwt["warmup"] = WarmUpLR(optimizer_kwt, total_iters=len(trainloadera) * config_kwt["hparams"]["scheduler"]["n_warmup"])

    if config_kwt["hparams"]["scheduler"]["scheduler_type"] is not None:
        total_iters = len(trainloadera) * max(1, (config_kwt["hparams"]["scheduler"]["max_epochs"] - config_kwt["hparams"]["scheduler"]["n_warmup"]))
        schedulers_kwt["scheduler"] = get_scheduler(optimizer_kwt, config_kwt["hparams"]["scheduler"]["scheduler_type"], total_iters)
    
    
    # Saving directory for the data2vec model
    config_d2v["exp"]["save_dir"] = os.path.join(config_d2v["exp"]["exp_dir"], config_d2v["exp"]["exp_name"])
    os.makedirs(config_d2v["exp"]["save_dir"], exist_ok=True)

    
    
    ######################################
    # train models
    ######################################
    
    
    step = 0
    best_acc = 0.0
    device = config_d2v["hparams"]["device"]
    log_file = os.path.join(config_d2v["exp"]["exp_dir"], "training_log.txt")
    best_avg_loss = 0.0
    n_batches = len(trainloaderf)
    
    for epoch in range(config_d2v["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss_d2v = 0.0
        running_target_var_d2v = 0.0
        running_prediction_var_d2v = 0.0
        running_loss_kwt = 0.0
        correct_kwt = 0
        
        for (dataf, targetsf), (dataa, targetsa) in zip(trainloaderf, trainloadera):
            batch_size = dataf.size(dim=0)
            audio_length = dataf.size(dim=-1)
            
            ###################################### 
            # data2vec step - friendly step
            ######################################
            
            # masking
            mask = mask_generator(shape=(batch_size, audio_length)).to(device)
            mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()
            
            # loading first K transformer layers of KWT
            kwt_partial_state_dict = load_partial_state_dict(model_kwt.state_dict(), k)
            model_d2v.load_state_dict(kwt_partial_state_dict, strict=False)
            
            # train single batch
            loss_d2v, target_var_d2v, prediction_var_d2v = data2vec.data2vec_utils.trainer.train_single_batch(model_d2v, dataf, mask, optimizer_d2v, criterion_d2v, device)
            model_d2v.ema_step()
            running_loss_d2v += loss_d2v
            running_target_var_d2v += target_var_d2v
            running_prediction_var_d2v += prediction_var_d2v
            
            # learning rate scheduler
            if schedulers_d2v["warmup"] is not None and epoch < config_d2v["hparams"]["scheduler"]["n_warmup"]:
                schedulers_d2v["warmup"].step()
            elif schedulers_d2v["scheduler"] is not None:
                schedulers_d2v["scheduler"].step()
            
            # logging data2vec step
            if not step % config_d2v["exp"]["log_freq"]:
                log_dict = {"epoch": epoch, "loss": loss_d2v, "lr": optimizer_d2v.param_groups[0]["lr"],
                            "target_var": target_var_d2v, "prediction_var": prediction_var_d2v}
                log(log_dict, step, config_d2v)
                
            
            ###################################### 
            # kwt step - adversarial step
            ######################################
            
            # loading first K layers of data2vec encoder
            d2v_partial_state_dict = load_partial_state_dict(model_d2v.state_dict(), k)
            model_kwt.load_state_dict(d2v_partial_state_dict, strict=False)
            
            # train single batch 
            loss_kwt, corr_kwt = utils.trainer.train_single_batch(model_kwt, dataa, targetsa, optimizer_kwt, criterion_kwt, device)
            running_loss_kwt += loss_kwt
            correct_kwt += corr_kwt
            
            # logging KWT step
            if schedulers_kwt["warmup"] is not None and epoch < config_kwt["hparams"]["scheduler"]["n_warmup"]:
                schedulers_kwt["warmup"].step()
            elif schedulers_kwt["scheduler"] is not None:
                schedulers_kwt["scheduler"].step()
            
            if not step % config_kwt["exp"]["log_freq"]:
                log_dict = {"epoch": epoch, "loss": loss_kwt, "lr": optimizer_kwt.param_groups[0]["lr"]}
            
            step += 1
            
        #################################################
        # epoch complete - log, validation and save model
        ################################################# 
        
        # data2vec log, validation and save model
        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0,
                    "avg_train_target_var": running_target_var_d2v / n_batches,
                    "avg_train_prediction_var": running_prediction_var_d2v / n_batches,
                    "avg_loss_per_ep": running_loss_d2v / len(trainloaderf.dataset)}
        log(log_dict, step, config_d2v)    
        
        if not epoch % config_d2v["exp"]["val_freq"]:
            avg_val_loss, avg_val_target_var, avg_val_prediction_var = data2vec.data2vec_utils.trainer.evaluate(model_d2v, mask_generator, criterion_d2v,
                                                                                valloaderf, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss,
                        "avg_val_target_var": avg_val_target_var, "avg_val_prediction_var": avg_val_prediction_var}
            #log(log_dict, step, config_d2v)
            
            # save best validation checkpoint
            if avg_val_loss < best_avg_loss or epoch == config_d2v["exp"]["val_freq"]:
                best_avg_loss = avg_val_loss
                save_path = os.path.join(config_d2v["exp"]["save_dir"], "best.pth")
                save_model(epoch, avg_val_loss, save_path, model_d2v, optimizer_d2v, log_file)
                save_path = os.path.join(config_d2v["exp"]["save_dir"], "best_encoder.pth")
                save_model(epoch, avg_val_loss, save_path, model_d2v.encoder, optimizer_d2v, log_file)
                
        # kwt log, validation and save model
        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0, "train_acc": correct_kwt/(len(trainloadera.dataset)), "avg_loss_per_ep": running_loss_kwt/len(trainloaderf)}
        log(log_dict, step, config_kwt)
        
        if not epoch % config_kwt["exp"]["val_freq"]:
            val_acc, avg_val_loss = utils.trainer.evaluate(model_kwt, criterion_kwt, valloadera, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            log(log_dict, step, config_kwt)
            
            # save best val ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config_kwt["exp"]["exp_dir"], "best.pth")
                save_model(epoch, val_acc, save_path, model_kwt, optimizer_kwt, log_file) 
    
    
    # training complete
    
    # data2vec evaluation 
    avg_val_loss, avg_val_target_var, avg_val_prediction_var = data2vec.data2vec_utils.trainer.evaluate(model_d2v, mask_generator, criterion_d2v, valloaderf,
                                                                        device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss,
                "avg_val_target_var": avg_val_target_var, "avg_val_prediction_var": avg_val_prediction_var}

    log(log_dict, step, config_d2v)

    # data2vec save final checkpoint
    save_path = os.path.join(config_d2v["exp"]["exp_dir"], "last.pth")
    save_model(epoch, avg_val_loss, save_path, model_d2v, optimizer_d2v, log_file)
    save_path = os.path.join(config_d2v["exp"]["exp_dir"], "last_encoder.pth")
    save_model(epoch, avg_val_loss, save_path, model_d2v.encoder, optimizer_d2v, log_file)
    
    # kwt evaluation
    
    val_acc, avg_val_loss = evaluate(model_kwt, criterion_kwt, valloadera, device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
    log(log_dict, step, config_kwt)

    # kwt save final checkpoint 
    save_path = os.path.join(config_kwt["exp"]["exp_dir"], "last.pth")
    save_model(epoch, val_acc, save_path, model_kwt, optimizer_kwt, log_file)
    
    
    
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
    config_kwt = get_config(args.confk)
    config_data2vec = get_config(args.confd)
    seed_everything(config_kwt['hparams']['seed'])
    alpha = 1
    adv_pretrain(config_kwt, config_data2vec, args.k, alpha)


if __name__ == "__main__":
    parser = ArgumentParser("Adversarial pretraining")
    parser.add_argument("--confk", type=str, required=True, help="Path to config.yaml file for KWT.")
    parser.add_argument("--confd", type=str, required=True, help="Path to config.yaml file for data2vec.")
    parser.add_argument("--k", type=int, required=True, help="First K transformer layers to update")
    args = parser.parse_args()
    main(args)