# JOANNA: Music Generation Using Weird Autoencoder and LSTM Architecture in Keras

------------------

I was able to generate music by training a NN model over Joanna Newsom's song "Sapokanikan" (here's the video for context and comparison, although in my opinion you should really listen to it anyways because it's gorgeous):

[![](http://img.youtube.com/vi/ky9Ro9pP2gc/0.jpg)](https://www.youtube.com/watch?v=ky9Ro9pP2gc "")

A 90 second sample with sample rate of 16000 hz from the song is overlappingly sliced into 2048 samples and then STFT'd, and use the frequency phase and magnitude as input. Then a 16 layer Denoising Autoencoder is trained with ELU activation followed by dropout layers for the encoder layers and linear activation for the decoder layers. I then put 2 LSTM layers in between the last encoder and first decoder (i.e. the deepest layer in the autoencoder). Each Autoencoder layer reduces the dimension by 250. I then train the LSTM model with overlapping encoded data. Unsurprisingly the LSTM layers add a lot more capacity into the autoencoder. 

To generate the model that will create new data, I feed the encoded data output from last encoder layer (i.e. before it's being processed by LSTM layers) onto layers of LSTM's, and then top it off with the LSTM layers from the autoencoder with frozen weights (i.e. untrainable). Instead of shifted blocks of input like I did [last time](https://www.reddit.com/r/MachineLearning/comments/3td04a/music_generation_using_stacked_denoising/), I instead use the shifted blocks of the output of the autoencoder after the LSTM layers.

I use 2 methods of generation: appended parts and whole generation. Appended parts appends the last block of each predictions into the results, while whole generation just use the whole output as the next input.

------------------

Part Generated:

[![](http://img.youtube.com/vi/zMD04EPm0mU/0.jpg)](https://www.youtube.com/watch?v=zMD04EPm0mU)

Whole Generated:

[![](http://img.youtube.com/vi/q0ZdSAkGo48/0.jpg)](https://www.youtube.com/watch?v=q0ZdSAkGo48)

I also tried different generation sequence length for comparison (listen with headphones):

[![](http://img.youtube.com/vi/_dwfxuLGsPA/0.jpg)](https://www.youtube.com/watch?v=_dwfxuLGsPA)

------------------

You need the following dependencies:

- Keras
- Theano
- numpy, scipy
- pyyaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.

------------------

Before you run this script, you need to open configure the script by opening config.py
```python
settings = {}
settings['source'] = './sources/sapo-160.wav' #this is the source wav material
settings['overlap'] = 2 #overlapping number for STFT
settings['coded-file'] = 'coded-file' #encoded output from autoencoder model
settings['lstm-file'] = 'lstm-file' #lstm model output
settings['phase-encoder'] = 'phase-encoder' #phase autoencoder weight file name
settings['magnitude-encoder'] = 'magnitude-encoder' #magnitude autoencoder weight file name
settings['phase-result'] = 'phase-result' #phase result file name
settings['magnitude-result'] = 'magnitude-result' #magnitude result file name
settings['lstm-epoch'] = 200
settings['ae-epoch'] = 200
settings['section-count'] = 8 #sample number
settings['ae-iteration'] = 40
settings['lstm-iteration'] = 30
settings['block-count'] = 175 #sample dimension
settings['layer-count'] = 4 #autoencoder layers count / 2
settings['sample-rate'] = 16000
settings['dim-decrease'] = 250 #dimensionality decrease for autoencoder
settings['load-weights'] = True #whether to load weights (usually for testing)
settings['dropout'] = 0.4 #dropout rate
```

------------------

After configuration, execute run.sh
