import cv2
import numpy as np
import pandas as pd

happy = cv2.imread('generated_data/emoji/happy.jpg')
sad = cv2.imread('generated_data/emoji/sad.jpg') 
angry = cv2.imread('generated_data/emoji/angry.jpg')
surprised = cv2.imread('generated_data/emoji/surprised.jpg')
neutral = cv2.imread('generated_data/emoji/neutral.jpg')
#pd dataframe that will be exported as my image labels
labels = pd.DataFrame(columns=['image', 'x', 'y', 'e_w', 'e_h','c_w', 'c_h', 'emotion'])
emotions = [happy, sad, angry, surprised, neutral]
emotion_label = ['happy', 'sad', 'angry', 'surprised', 'neutral']

for i in range(20):
    canvas_x = np.random.randint(200, 1000)
    canvas_y = np.random.randint(200, 1000)
    #white canvas
    canvas = np.ones((canvas_x, canvas_y, 3), np.uint8) * 255
    for j in range(1):
        #choose random emotion
        selected_emotion = np.random.randint(0, 5)
        emotion = emotions[selected_emotion]
        #random resize and rotate the emotion
        emotion = cv2.resize(emotion, (np.random.randint(50, 200), np.random.randint(50, 200)))
        emotion = cv2.rotate(emotion, np.random.randint(0, 3))

        mask = np.all(emotion < 250, axis=2)
        color = np.random.randint(0, 256, 3).tolist()
        for c in range(3):
            emotion[:,:,c][mask] = color[c]
        #random placement
        x = np.random.randint(0, canvas_x - emotion.shape[0])
        y = np.random.randint(0, canvas_y - emotion.shape[1])
        mask = np.all(emotion < 250, axis=2)
        
        # Apply the emotion only where the mask is True
        for c in range(3):  # For each color channel
            canvas[x:x+emotion.shape[0], y:y+emotion.shape[1], c][mask] = emotion[:,:,c][mask]
    cv2.imwrite(f'generated_data/image/img_{i}.jpg', canvas)
    labels.loc[-1] = [f'img_{i}.jpg', x, y, emotion.shape[0], emotion.shape[1], canvas_x, canvas_y, emotion_label[selected_emotion]]
    labels.index = labels.index + 1
labels = labels.sort_index()
labels.to_csv('generated_data/labels.csv')