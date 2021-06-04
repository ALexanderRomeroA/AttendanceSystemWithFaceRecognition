import shutil
import random
import numpy
import os
#to prepare data for training and testing

os.chdir('data')
dir=os.curdir
if os.path.isdir('train') is False:
    os.mkdir('train')
    os.mkdir('test')
    print(os.curdir)
    for i in os.listdir(dir):
        shutil.move(f'{i}','train')
        os.mkdir(f'test/{i}')
        print(os.listdir(f'train/{i}'))
        test_sample=random.sample(os.listdir(f'train/{i}'),5)
        
        for j in test_sample:
            shutil.move(f'train/{i}/{j}',f'test/{i}')

os.chdir('../')