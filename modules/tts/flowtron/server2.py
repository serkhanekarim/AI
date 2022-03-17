###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import argparse
import json
import sys
import numpy as np
import torch
import sounddevice as sd

from flowtron import Flowtron
from torch.utils.data import DataLoader
from data_en import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write

import timeit

seed = 1234
waveglow_path = '/home/serkhane/models/tts/waveglow/waveglow_256channels_universal_v5.pt'
flowtron_path = '/home/serkhane/models/tts/flowtron/flowtron_ljs.pt'
speaker_id = 0
gate_threshold = 0.5
n_frames = 400
sigma = 0.5


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

with open('config/config_ljs.json') as f:
    data = f.read()

global config
config = json.loads(data)


data_config = config["data_config"]
global model_config
model_config = config["model_config"]


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
stream = sd.OutputStream(samplerate=22050, channels=1)





# load waveglow
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
waveglow.cuda()
for k in waveglow.convinv:
    k.float()
waveglow.eval()

# load flowtron
model = Flowtron(**model_config).cuda()
checkpoint = torch.load(flowtron_path, map_location='cpu')
if 'model' in checkpoint:
    state_dict = checkpoint['model'].state_dict()
else:
    state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.eval()
print("Loaded checkpoint '{}')" .format(flowtron_path))


ignore_keys = ['training_files', 'validation_files']
trainset = Data(
    data_config['training_files'],
    **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))
speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
speaker_vecs = speaker_vecs[None]

while True:
    text = input("Text >>")
    tic = timeit.default_timer()
    if text[-1] not in ["?", ".", "!", ",", ";", ":"]: text += "."
    text = text.capitalize()
    n_frames = max(50,len(text) * 6)
    #n_frames = len(text) * 6
    text = trainset.get_text(text).cuda()
    text = text[None]

    stream.start()

    with torch.no_grad():
        residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
        mels, attentions = model.infer(
            residual, speaker_vecs, text, gate_threshold=gate_threshold)

    with torch.no_grad():
        audio = waveglow.infer(mels, sigma=0.8).float()

    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    toc = timeit.default_timer()
    print("Processing duration: " + str(toc - tic))
    print(audio.shape)

    stream.write(audio)
    stream.stop()
