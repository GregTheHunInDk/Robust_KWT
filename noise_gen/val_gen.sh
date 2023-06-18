#!/bin/bash

python3 noise_gen_validation.py --dp home/ubuntu/vol1/GoogleSpeechCommands/speech_commands_v0.02/ home/ubuntu/vol1/GoogleSpeechCommands/noise/kolbek_slt2016/bbl/bbl_val.wav --snr -5
python3 noise_gen_validation.py --dp home/ubuntu/vol1/GoogleSpeechCommands/speech_commands_v0.02/ home/ubuntu/vol1/GoogleSpeechCommands/noise/kolbek_slt2016/bus/bus_val.wav --snr -5
python3 noise_gen_validation.py --dp home/ubuntu/vol1/GoogleSpeechCommands/speech_commands_v0.02/ home/ubuntu/vol1/GoogleSpeechCommands/noise/kolbek_slt2016/caf/caf_val.wav --snr -5
python3 noise_gen_validation.py --dp home/ubuntu/vol1/GoogleSpeechCommands/speech_commands_v0.02/ home/ubuntu/vol1/GoogleSpeechCommands/noise/kolbek_slt2016/ped/ped_val.wav --snr -5
python3 noise_gen_validation.py --dp home/ubuntu/vol1/GoogleSpeechCommands/speech_commands_v0.02/ home/ubuntu/vol1/GoogleSpeechCommands/noise/kolbek_slt2016/ssn/ssn_val.wav --snr -5