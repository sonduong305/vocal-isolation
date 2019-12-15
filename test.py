from model import VocalModel
from predict import predict_song, to_int_wav, predict_from_youtube
import librosa
import time
import torch
from youtube_download import get_audio
from pydub import AudioSegment
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from torchsummary import summary
import stempeg
import numpy as np

def extract_vocal(fname):
    stems, _ = stempeg.read_stems(fname, stem_id=[0,4])
    stems = stems.astype(np.float32)

    master = stems[0,:,:]
    vocal = stems[1,:,:]
    
    return master, vocal
def load_file(path):
    
    song = AudioSegment.from_file(path, sample_width = 2, channels = 1, frame_rate= 44100)
    song = song.get_array_of_samples()
    song = np.array(song)
    song = np.array(song / 32768, dtype = np.float)
    return song


model = VocalModel()

model.cuda()
# print(model)
model.load_state_dict(torch.load('models\\torch_model_v1.pth'))


start = time.time()
# song, sr = librosa.load('data\\master.wav', 44100, mono=True)
# song = predict_song(song, model)
# plt.figure(figsize=(8, 4))
# # librosa.display.waveplot(song[1000000:1003000], sr = 44100)
# S = librosa.feature.melspectrogram(y=song[4000000:4500000], sr=sr, n_mels=128,fmax=8000)
# S_dB = librosa.power_to_db(S, ref=np.max)
# librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr, fmax=8000)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Vocal Groud Truth')
# plt.show()
# print(song.shape)

# print(get_audio("https://www.youtube.com/watch?v=fMIEdGtEyeo"))
predict_from_youtube("https://www.youtube.com/watch?v=RM4IlbrnUHQ", model)

print(time.time() - start)
# master, vocal = extract_vocal('data\\train\\The Long Wait - Back Home To Blue.stem.mp4')
# # librosa.output.write_wav('data\\predicted.wav', song, 44100)
# librosa.output.write_wav('data\\master.wav', master, 44100)
# librosa.output.write_wav('data\\vocal.wav', vocal, 44100)