#!/bin/bash

python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/str/str_val.wav --snr -5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/bbl/str_val.wav --snr -5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/bus/str_val.wav --snr -5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/caf/str_val.wav --snr -5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ped/str_val.wav --snr -5
python3 noise_gen_validation.py --dp ~/GoogleSpeechCommands/speech_commands_v0.02/ --np ~/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/str_val.wav --snr -5