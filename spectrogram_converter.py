import librosa
import librosa.display
import cv2
import musdb
import numpy as np

mus = musdb.DB(root = "D:\\ws\\senior-project\\data\\")

for track in mus:
    # train(track.audio, track.targets['vocals'].audio)
    print(track.name)
    X = librosa.stft(librosa.to_mono(track.audio.T))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(X), ref=np.max), y_axis='log', x_axis='time')
    # print(X[:, 300:310])
#     X = np.array(X[:, :1000], np.uint8)
#     cv2.imshow("spec", X)
# cv2.waitKey()
