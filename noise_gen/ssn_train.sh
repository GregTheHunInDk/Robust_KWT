#!/bin/bash

python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr -10
python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr -5
python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr 5
python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr 0
python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr 5
python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr 10
python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr 15
python3 noise_gen_training.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_train.wav --snr 20

python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr -10
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr -5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr 5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr 0
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr 5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr 10
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr 15
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr 20