import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import random
import os.path


def gen_val_samples():
    val_filenames = load_filename(parse_data_cfg(opt.data)['valid'])
    out, num_images = opt.output, opt.num
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    val_filenames_sample = random.sample(val_filenames, num_images)

    for index in range(len(val_filenames_sample)):
        current_val_filenames_sample_full = val_filenames_sample[index]
        current_val_filenames_sample = os.path.basename(current_val_filenames_sample_full)
        shutil.copyfile(current_val_filenames_sample_full, out + '/' + current_val_filenames_sample)
        print('%s <--> %d/%d' % (current_val_filenames_sample, index, num_images))

    print('Done')

def load_filename(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/high-speed-yolov3-20191030.data', help='coco.data file path')
    parser.add_argument('--output', type=str, default='data/val-samples', help='output folder')  # output folder
    parser.add_argument('--num', type=int, default='30', help='num of images to test')  # output folder
    opt = parser.parse_args()
    print(opt)

    gen_val_samples()
