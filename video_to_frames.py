import cv2
import numpy as np
import os
import time
from tqdm import tqdm

files = os.listdir('./faceforensics++/manipulated_sequences')
shortNames = {'Deepfakes': 'df_',
              'Face2Face': 'ff_',
              'FaceSwap': 'fs_',
              'NeuralTextures': 'nt_'}

for midFileName in files:

    # set video file path of input video with name and extension
    arr = sorted(list(os.listdir(f'./faceforensics++/manipulated_sequences/{midFileName}/c23/videos/')))


    frame_skip = 10
    index = 0

    for i in tqdm(arr[:40]):
        st = f'./faceforensics++/manipulated_sequences/{midFileName}/c23/videos/' + i
        vid = cv2.VideoCapture(st)
        if not os.path.exists(f'./frames/{midFileName}'):
            os.makedirs(f'./frames/{midFileName}')

        # for frame identity
        frame_count = 0
        while True:
            # Extract images
            ret, frame = vid.read()
            # end of frames
            if not ret:
                break
            # Saves images

            if frame_count == frame_skip:
                break
            else:
                name = f'./frames/{midFileName}/' + f'{shortNames[midFileName]}' + str(index) + '.jpg'
                cv2.imwrite(name, frame)
                frame_count += 1

            # next frame
            index += 1

