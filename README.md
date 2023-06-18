This repository contains the adversarial networks of robust KWT. The project is the continuation of https://github.com/HolgerBovbjerg/data2vec-KWS. 

# Baseline models
The system requires Python 3.8. The codes are not compatible with newer versions of Python. The codes can be run on Ubuntu 18.04, Windows 10 and above, and macOS. The codes were observed to run with error on newer versions of Ubuntu. 
To set up the baseline models, first the dependencies must be installed.  
For that the following has the run: 
``` shell
pip install -r requirements.txt
```
Then the Google Speech Commands can be downloaded as: 
``` shell
sh download_gspeech_v2.sh
``` 
The reduced training dataset can be achieved by running: 
```shell
python make_data_list.py --pretrain <amount_of_train_set_for_pretrain> -v <path/to/validation_list.txt> -t <path/to/testing_list.txt> -d <path/to/dataset/root> -o <output dir>
```

The config files can be found in the KWT_configs and the data2vec/data2vec_config files. 
The following command runs the supervised training of the KWT: 
``` shell 
python3.8 train.py --conf KWT_configs/<name_of_the_config_file>.yaml
``` 
To pretrain the models in the data2vec system, the following code must be run: 
``` shell
python3.8 train_data2vec.py --conf data2vec/data2vec_configs/<name_of_the_config_file>.yaml
```
# Adversarial Training
The adversarial training can be run by: 
``` shell
python3.8 adv_train.py --conf adv_configs/finetune/clean/<snr_version>/<config_name>/<number_of_shared_layers>.yaml
```
The adversarial pretraining is similar, but it requires two config files (the value of alpha is fixed in the code): 
``` shell
python3.8 adv_pretrain.py --confk adv_configs/petrain/KWT/<name_of_the_config_file>.yaml --confd adv_configs/petrain/data2vec/<name_of_the_config_file>.yaml --k <number_of_shared_layers>
```
# Noisy data generation
The noisy dataset generator codes can be found in the noise_gen folder. 
To run test data with custom noise, the following must be run: 
``` shell
python3.8 noise_gen.py --dp <path_to_the_test_data> --np <path_to_the_noise.wav> --snr <signal_to_noise_ratio_in_dB>
```
The noise_gen_training.py and noise_gen_validation.py generate noisy training and validation data. 
In order to generate noisy training data for adversarial training the following code can be run: 
``` shell
python3.8 mixed_train_adv.py --snr <snr> 
```
The noise is saved in the following path: /home/ubuntu/noisy_speech_commands_v0.02/seeded/adv/

