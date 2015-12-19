#!/bin/sh

python finetuning-autoencoder.py
nvidia-smi
python encoder.py
nvidia-smi
python coded-lstm-trainer.py
nvidia-smi
python encoder-lstm-decoder-full.py
nvidia-smi
python decode-waveform.py
nvidia-smi
