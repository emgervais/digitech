import PIL.Image as pil
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import cv2
from sliding import match_template

def match_emotion(emoji):
    happy = np.array(pil.open('data/emojis/happy.jpg').convert('L'))
    sad = np.array(pil.open('data/emojis/sad.jpg').convert('L'))
    angry = np.array(pil.open('data/emojis/angry.jpg').convert('L'))
    surprised = np.array(pil.open('data/emojis/surprised.jpg').convert('L'))
    crying = np.array(pil.open('data/emojis/crying.jpg').convert('L'))
    emotions = [happy, sad, angry, surprised, crying]
    emotions_name = ['happy', 'sad', 'angry', 'surprised', 'crying']
    result = []
    for i in range(len(emotions)):
        result.append(match_template(emoji, emotions[i], emoji=True)[1])
    max_pos = np.argmax(result)
    return emotions_name[max_pos]

def bonus_find_emoji(path="data/train/dataset/emoji_1113.jpg"):
    image =  np.array(pil.open(path).convert('L'))
    #hough circle
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=35,
                                   minRadius=10, maxRadius=50)
    if circles is None:
        # print("no emoji found")
        return

    circles = np.round(circles[0, :]).astype("int")
    boxes = []
    padding = 10

    for (x, y, r) in circles:
        extended_r = r + padding
        real_x, real_y = x - extended_r, y - extended_r
        patch = image[y-extended_r:y+extended_r, x-extended_r:x+extended_r]
        boxes.append([[real_x, real_y, r], patch])

    for ((real_x, real_y, r), patch) in boxes:
        template = np.array(pil.open('data/emojis/happy.jpg').convert('L'))
        result = match_template(patch, template)[0]
        x_match, y_match = real_x + result[0], real_y + result[1]
        emoji = image[y_match:y_match + r * 2, x_match:x_match + r * 2]
        emotion = match_emotion(emoji)
        print(f'Emoji: {emotion} Coordinates: ({x_match}, {y_match})')
    # cv2.rectangle(image, (real_x + result[0], real_y + result[1]), (real_x + result[0] + 50, real_y + result[1] + 50), (0, 255, 0), 4)
    # plt.imshow(image)
    # plt.savefig("a.png")

def guess():
    labels = pd.read_csv('data/bonus/labels.csv', delimiter=';')
    for i in range(len(labels)):
        # if i == 50:
        #     break
        try:
            print('Picture:', labels['file_name'][i])
            bonus_find_emoji(path=f'data/bonus/dataset/{labels['file_name'][i]}')
        except:
            continue

guess()