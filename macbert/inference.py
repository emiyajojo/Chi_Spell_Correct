import os
import sys
sys.path.append('../..')
from config import Args
from utils import Inference
import time

def func_macbert4csc_inference(model, input_text):
    start_time = time.time()
    result = model.predict(input_text)
    end_time = time.time()
    print('单条样本推理耗时: {}s'.format(end_time - start_time))
    return result

if __name__ == '__main__':
    s = 'X'
    model = Inference()
    while True and s != 'q' and s != 'Q':
        person_input = input('Please input a sentence:')
        res = func_macbert4csc_inference(model, person_input)
        print('res = ', res)
        s = input('Continue or not (q/Q)')
