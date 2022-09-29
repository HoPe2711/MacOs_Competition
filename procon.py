import numpy as np
import librosa
from tqdm import tqdm
import json
import requests
from urllib.request import urlretrieve
import tensorflow as tf

TOKEN = '5504d2574a055c7fee479f5c5d6f4c14dc7117c88736fcf9e4d7393815a38d32'
HOST = 'https://procon33-practice.kosen.work'
RATE = 48000

file_name = './audio_input/'
headers = {'Procon-Token': TOKEN}
num_problems = 0
num_data = 0
list_problems = []
curr_problem = ""
problem_answer = []


def get_melspectrogram(filename):
    y, sr = librosa.load(filename, sr=RATE)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    mel_spect = mel_spect / 80 + 1
    return mel_spect


model = tf.keras.applications.EfficientNetV2B0(include_top=True, weights=None, 
                                               input_shape=(128, 128, 1), classes=88,
                                               classifier_activation='sigmoid',
                                               include_preprocessing=False)
model.summary()
model.load_weights("./weight-model.h5")

def get_result(idx):
    res = []
    idx = sorted(idx)
    for i in idx:
        if i >= 44:
            if i-44+1 < 10:
                res.append("0" + str(i-44+1))
            else: 
                res.append(str(i-44+1))
        else: 
            if i+1 < 10:
                res.append("0" + str(i+1))
            else: 
                res.append(str(i+1))
    return res

def predict(file):
    x_test = get_melspectrogram(file)
    x_test = x_test.transpose()
    x_test = np.expand_dims(x_test, -1)
    y_pred = model.call(tf.convert_to_tensor(np.expand_dims(x_test, 0)))
    return y_pred.numpy()[0]

def gm():
    get_match = requests.get(f'{HOST}/match', headers=headers)
    global num_problems 
    global list_problems 
    list_problems = []
    try:
        num_problems =  get_match.json()['problems']
    except:
        print('Access time')
    return get_match.text

def gp():
    get_problem = requests.get(f'{HOST}/problem', headers=headers)
    
    try:
        global num_data 
        global curr_problem
        num_data =  get_problem.json()['data']
        curr_problem = get_problem.json()['id']
        print(num_data)
    except:
        print('Access time')
    return get_problem.text

def download(url, file_name):
    with open(file_name, "w") as file:
        response = requests.get(url)
        file.write(response.content)

def gc(n):
    get_chunk = requests.post(f'{HOST}/problem/chunks?n={n}', headers=headers)
    ans = np.ones(88)
    try:
        for i in get_chunk.json()['chunks']:
            file = file_name + i
            urlretrieve(f'{HOST}/problem/chunks/{i}?token={TOKEN}' , file)
            if file != None:
                ans = np.multiply(ans, predict(file))
        idx = np.argsort(ans)[-num_data:]
        global problem_answer
        problem_answer = get_result(idx)
        print(problem_answer)
    except:
        print('Access time')
    return get_chunk.text

def sb():
    answer = {
        "problem_id": curr_problem,   
        "answers": problem_answer
    }
    answer = json.dumps(answer)
    submit_answer = requests.post(f'{HOST}/problem', data=answer, headers=headers)
    return submit_answer.text

while True:
    inp = input()
    print(eval(inp))