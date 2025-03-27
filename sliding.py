import PIL.Image as pil
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure

np.set_printoptions(threshold=sys.maxsize)

def detect_peaks(image, threshold=0.90, min_distance=10):
    struct = generate_binary_structure(2, 2)
    
    local_max = maximum_filter(image, footprint=struct) == image
    
    threshold_peaks = image > threshold * np.max(image)
    
    detected_peaks = local_max & threshold_peaks
    
    coordinates = np.where(detected_peaks)
    
    if len(coordinates[0]) == 0:
        return np.array([])
    
    peaks = np.column_stack((coordinates[1], coordinates[0]))
    
    if min_distance > 0:
        peak_values = [image[y, x] for x, y in peaks]
        sorted_indices = np.argsort(peak_values)[::-1]
        peaks = peaks[sorted_indices]
        
        mask = np.ones(len(peaks), dtype=bool)
        
        for i in range(len(peaks)):
            if mask[i]:
                x, y = peaks[i]
                
                distances = np.sqrt(np.sum((peaks - np.array([x, y]))**2, axis=1))
                
                close_points = np.where(distances < min_distance)[0]
                mask[close_points[close_points > i]] = False
        
        peaks = peaks[mask]
    
    return peaks

def match_template(img, template):
    h, w = template.shape
    H, W = img.shape
    
    _, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    search_rect = Rectangle((0, 0), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(search_rect)
    plt.title('Template Matching Process')
    plt.ion()
    plt.show()
    
    pad_h = H - h
    pad_w = W - w
  
    t_mean = np.mean(template)
    t_std = np.std(template)
    t_norm = template - t_mean
    t_norm_padded = np.pad(t_norm, ((0, pad_h), (0, pad_w)), mode='constant')

    img_fft = np.fft.fft2(img)
    t_fft = np.fft.fft2(t_norm_padded)
  
    result = np.fft.ifft2(img_fft * np.conj(t_fft)).real
  
    img_sq = img**2
    img_sum = np.zeros_like(result)
    img_sum_sq = np.zeros_like(result)
  
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    integral_sq = np.cumsum(np.cumsum(img_sq, axis=0), axis=1)
    
    vis_step = 100
    
    for y in range(13, result.shape[0] - h + 1):
        for x in range(result.shape[1] - w + 1):
            if y > 0 and x > 0:
                img_sum[y, x] = integral[y+h-1, x+w-1] - integral[y-1, x+w-1] - integral[y+h-1, x-1] + integral[y-1, x-1]
                img_sum_sq[y, x] = integral_sq[y+h-1, x+w-1] - integral_sq[y-1, x+w-1] - integral_sq[y+h-1, x-1] + integral_sq[y-1, x-1]
            elif y > 0:
                img_sum[y, x] = integral[y+h-1, x+w-1] - integral[y-1, x+w-1]
                img_sum_sq[y, x] = integral_sq[y+h-1, x+w-1] - integral_sq[y-1, x+w-1]
            elif x > 0:
                img_sum[y, x] = integral[y+h-1, x+w-1] - integral[y+h-1, x-1]
                img_sum_sq[y, x] = integral_sq[y+h-1, x+w-1] - integral_sq[y+h-1, x-1]
            else:
                img_sum[y, x] = integral[y+h-1, x+w-1]
                img_sum_sq[y, x] = integral_sq[y+h-1, x+w-1]
                
            if x % vis_step == 0 and y % vis_step == 0:
                search_rect.set_xy((x, y))
                plt.title(f"Searching at position ({x}, {y})")
                plt.draw()
                plt.pause(0.001)
  
    n = h * w
    patch_means = img_sum / n
    patch_stds = np.sqrt((img_sum_sq / n) - patch_means**2)
  

    mask = patch_stds > 0
    result[mask] = result[mask] / (patch_stds[mask] * t_std * n)
    
    matches = detect_peaks(result)
    
    search_rect.remove()
    
    for match in matches:
        x, y = match
        match_rect = Rectangle((x, y), w, h, 
                            linewidth=3, edgecolor='g', facecolor='none')
        ax.add_patch(match_rect)
        
    plt.title(f"Found {len(matches)} matches")
        
    plt.draw()
    plt.pause(2)
    plt.ioff()
    
    return matches

def find_emoji(img_path, emoji_path):
    img = np.array(pil.open(img_path).convert('L'))
    emoji = np.array(pil.open(emoji_path).convert('L'))
    
    result = match_template(img, emoji)
    return result
def main(img_path=None, emoji_path='data/basic/dataset/emoji.jpg'):
    label = pd.read_csv('data/basic/labels.csv', delimiter=';')
    i = 0
    for i in range(len(label)):
        file_name = label['file_name'][i]
        img_path = 'data/basic/dataset/' + file_name
        matches = find_emoji(img_path, emoji_path)[0]
        print(f"Matches for {file_name}: {matches[0] == int(label['x_s'][i][1:-1]) and matches[1] == int(label['y_s'][i][1:-1])}")

if __name__ == '__main__':
    main()