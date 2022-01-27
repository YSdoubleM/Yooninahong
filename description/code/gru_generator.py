import numpy as np
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('/Users/hongseongmi/Documents/git_ws/Yooninahong/description/senten_generating_model2.h5')

# load vec ID
with open('/Users/hongseongmi/Documents/git_ws/Yooninahong/description/data/char2idx2.pickle', 'rb') as fr:
    char2idx = pickle.load(fr)

n = 35
max_len = 30
total = []
objects = {
    'person':'사람',
    'person_back':'사람',
    'car':'차',
    'bike':'자전거',
    'motorcycle':'오토바이',
    'electricscooter':'전동킥보드',
    'bollard':'볼라드'
    }

def main(key):
    key = list(key.split())
    length = len(key)
    obj = [objects[key[i]] for i in range(1, length, 2)]
    generate(obj)
    # for i in range(0, length, 2):
    #     num = int(key[i])
    #     if num > 1:
    #         pass
    #     else:
    #         pass

def generate(obj):
    olen = len(obj)
    print(olen)

    for o in obj:
        current_word = o
        init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
        sentence = ''
        for _ in range(n): # n번 반복
            encoded = [char2idx[token] for token in current_word]#t.texts_to_sequences([current_word])[0] 
            # 현재 단어에 대한 정수 인코딩
            
            encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre') 
            # 데이터에 대한 패딩
            
            result = np.argmax(model.predict(encoded), axis=-1) 
            
            # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
            for word, index in char2idx.items(): 
                if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                    break # 해당 단어가 예측 단어이므로 break
            current_word = current_word + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
            sentence = sentence + word # 예측 단어를 문장에 저장
            if word == '<EOS>':
                break
        # for문이므로 이 행동을 다시 반복
        sentence = init_word + sentence
        if olen > 1:
            sentence = sentence.replace('습니다<EOS>',"고 ")
            olen -= 1
        else:
            sentence = sentence.replace('<EOS>',"")
        print(sentence)
        total.append(sentence)

    return ''.join(total)
