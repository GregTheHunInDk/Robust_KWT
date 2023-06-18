import numpy as np
import librosa
import os
import shutil
from argparse import ArgumentParser
import soundfile as sf
from tqdm import tqdm

"""
This is a script that generates noisy data using FaNT. 
"""


# setting up arguments
parser = ArgumentParser(description="Arguments for generating noisy validation data")
#parser.add_argument('--dp', type=str, required=True, help="Main folder that contains the validation-test data")
#parser.add_argument('--np', type=str, required=True, help="Path of the noise file")
#parser.add_argument('--sd', type=str, required=True, help="Path to save the new generated/sorted data")
parser.add_argument('--snr', type=str, required=True, help="Signal to noise ration in dB")

args = parser.parse_args()


def main(args):
    # noise path: current_path/noise_type/snr/data_types[]/
    # data path: ~/.../data_path
    data_path = "/home/ubuntu/speech_commands_v0.02/"
    noise_path = [
        "/home/ubuntu/noise/bbl/bbl_val.wav", 
        "/home/ubuntu/noise/bus/bus_val.wav",
        "/home/ubuntu/noise/caf/caf_val.wav",
        "/home/ubuntu/noise/str/str_val.wav"
    ]
    noise_names = ['bbl', 'bus', 'caf', 'str', 'cln']
    snr = args.snr
    save_path = '/home/ubuntu/noisy_speech_commands_v0.02/seeded/adv'
    data_list = data_path + "_generated/sorted/validation_list.txt"
    print(data_list)
    data_list = "/home/ubuntu/speech_commands_v0.02/_generated/sorted/validation_list.txt"
    noise_name = "mixed" #noise_path.split('/')[-1][0:-4] # takes the last part of the path (file name) and removes the .wav extention
    num_noises = len(noise_names)
    current_path = os.getcwd()

    with open(data_list) as dlist:
        datas = dlist.read().splitlines()

    
    # Creating the path for the noisy data
    noisy_path = '/validation_' + noise_name + '/snr_' + snr + '/'
    input_raw_path = '/validation_' + noise_name +  '/snr_' + snr + '/input_raw/'
    output_raw_path = '/validation_' + noise_name + '/snr_' + snr + '/output_raw/'
    print("################# Path info #################\n")
    print(f'Data list: {data_list}')
    print(f'Current directory: {current_path}')
    print(f'Length of the data list: {len(datas)}')
    print(f'Data path: {data_path}')
    print(f'Noise path: {noise_path}')
    print(f'Noise name: {noise_name}')
    print(f'Noisy data saving path: {noisy_path}')
    print(f'Clean raw data path: {input_raw_path}')
    print("\n############################################")
    #if not os.path.exists(current_path + noisy_path):

    datas_full = [os.path.join(data_path, data) for data in datas] # data list extended with current directory
    if not os.path.exists(current_path + noisy_path):
        os.makedirs(current_path + noisy_path)
    in_list = open(current_path + noisy_path + "in.list", 'a+')
    out_list = open(current_path + noisy_path + "out.list", 'a+')
    label_list = open(current_path + noisy_path + "labels.txt", 'a+')
    noise_labels = int(len(datas)/len(noise_path) + 1) * [i for i in range(len(noise_path))]
    noise_labels = noise_labels[0:len(datas)]

    print("\nGenerating in and out list for FaNT\n")
    for i, data in tqdm(enumerate(datas)):
        # loading data with librosa
        data_tr = data.split('/')[-2] + '/' + data.split('/')[-1]
        x, _ = librosa.load(datas_full[i], sr=None)
        x *= 2**(15-1)
        x = x.astype(np.int16)
        #label_list.write(str(noise_labels[i]) + "\n")
        # creating the input (clean) .raw directory
        if not os.path.exists(current_path + input_raw_path + data_tr.split('/')[0]):
            print(current_path + input_raw_path + data_tr.split('/')[0], end='\r')
            os.makedirs(current_path + input_raw_path + data_tr.split('/')[0] + '/')
        # creating file if not exists and addigin it to the input and output list
        if not (os.path.exists(current_path + input_raw_path + data_tr.split('.')[-2] + ".raw")):
            fid_in = open(current_path + input_raw_path + data_tr.split('.')[-2] + ".raw", 'wb')
            fid_in.write(x)
            in_list.write(input_raw_path[1:] + data_tr.split('.')[-2] + ".raw\n")
            out_list.write(output_raw_path[1:] + data_tr.split('.')[-2] + '_' + str(noise_names[i % num_noises]) + ".raw\n")
            fid_in.close()
        # creating the output (noisy) .raw directory
        if not os.path.exists(current_path + output_raw_path + data_tr.split('/')[0]):
            os.makedirs(current_path + output_raw_path + data_tr.split('/')[0] + '/')
    
    in_list.close()
    out_list.close()
    
    in_list = open(current_path + noisy_path + "in.list", 'r')
    out_list = open(current_path + noisy_path + "out.list", 'r')
    
    print("\nGenerating raw files for data contamination...\n")   
    
    in_lines = in_list.readlines()
    out_lines = out_list.readlines()
    
    print("in_list length: ", len(in_lines))
    print("out_list length: ", len(out_lines))
    print("noise_labels length: ", len(noise_labels))
    print("\n")
    print("################# Running FaNT #################\n")
    
    random_seed_iterator = 0
    for in_data, out_data, noise_label in tqdm(zip(in_lines, out_lines, noise_labels)):
        in_sample = open(current_path + noisy_path + "in_sample.list", 'w+')
        out_sample = open(current_path + noisy_path + "out_sample.list", 'w+')
        in_sample.write(in_data)
        out_sample.write(out_data)
        in_sample.close()
        out_sample.close()
        fant_command = f'./filter_add_noise -i {current_path + noisy_path + "in_sample.list"} -o {current_path + noisy_path + "out_sample.list"} -n {noise_path[noise_label]} -u -m snr_8khz -r {random_seed_iterator} -d -s {snr}'
        os.system(fant_command)
        random_seed_iterator += 1
     
    out_list = open(current_path + noisy_path + "out.list", 'r')
    out_paths = out_list.readlines()
    out_paths = [current_path + '/' + out_path[:-1] for out_path in out_paths]
    idx = 0
    
    if not os.path.exists(save_path + noisy_path):
        os.makedirs(save_path + noisy_path)
    noisy_data_list = open(save_path + noisy_path + "validation_list.txt", "w")
    
    print("Writing .wav files...")
    file_path = out_paths[0].split('/')[-2] + '/' + out_paths[0].split('/')[-1]
    
    for out_path in tqdm(out_paths):
        out_file = open(out_path, 'rb')
        x = out_file.read()
        x = np.frombuffer(x, dtype=np.int16)
        out_file.close()
        x = x / 2**15
        x = x.astype(np.float32)
        file_name = out_path.split('/')[-2] + '/' + out_path.split('/')[-1][:-4]
        idx += 1
        #print(f' {idx} out of {len(out_paths)}', end='\r')
        if not os.path.exists(save_path + noisy_path + file_name.split('/')[0]):
            os.makedirs(save_path + noisy_path + file_name.split('/')[0])
        sf.write(save_path + noisy_path + file_name + '.wav', x, 16000, 'PCM_16')
        noisy_data_list.write(save_path + noisy_path + file_name + ".wav\n")
    
    noisy_data_list.close()
    
    print("\nDone.")

main(args)
