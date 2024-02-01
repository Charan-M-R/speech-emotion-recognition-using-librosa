import pickle
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from feature_extract import extract_feature
simplefilter(action='ignore', category=FutureWarning)

modelfinal=pickle.load(open('audiomodelfinal.pkl','rb'))

from pydub import AudioSegment
from pydub.silence import split_on_silence
#model=pickle.load(open('audiomodel.pkl','rb'))

sound_file = AudioSegment.from_wav("D:/6th Sem/AIML/Project(1)/AudioPredict/Audio/testaudio1.wav")
audio_chunks = split_on_silence(sound_file, min_silence_len=3000, silence_thresh=-50)
 
for i, chunk in enumerate(audio_chunks):
   out_file = "chunk{0}.wav".format(i)
   print("exporting", out_file)
   chunk.export(out_file, format="wav")
print()



# predict emotion for each audio chunk

for file in glob.glob("D:/6th Sem/AIML/Project(1)/AudioPredict/chunk*.wav"):
    #print(file)
    x1= []
    feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
    x1.append(feature)
    try_test=np.array(x1)
    emo_pred=modelfinal.predict(try_test)
    print("Predicted Emotion :"+ emo_pred[0])