import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import itertools

def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol].imshow(img_array[i])

def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))
    plt.title('{}'.format(title))
    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))
    plt.show()

def get_cityscapes_palette():
    """
    Returns the color palette for Cityscapes classes as a numpy array.
    This is based on the CARLA simulator's semantic segmentation colors.
    """
    return np.array([
        [0, 0, 0],         # 0: unlabeled
        [128, 64, 128],    # 1: road
        [244, 35, 232],    # 2: sidewalk
        [70, 70, 70],      # 3: building
        [102, 102, 156],   # 4: wall
        [190, 153, 153],   # 5: fence
        [153, 153, 153],   # 6: pole
        [250, 170, 30],    # 7: traffic light
        [220, 220, 0],     # 8: traffic sign
        [107, 142, 35],    # 9: vegetation
        [152, 251, 152],   # 10: terrain
        [70, 130, 180],    # 11: pedestrian
        [220, 20, 60],     # 12: sky
        [255, 0, 0],       # 13: rider
        [0, 0, 142],       # 14: car
        [0, 0, 70],        # 15: truck
        [0, 60, 100],      # 16: bus
        [0, 80, 100],      # 17: train
        [0, 0, 230],       # 18: motorcycle
        [119, 11, 32],     # 19: bicycle
        [110, 190, 160],   # 20: static
        [170, 120, 50],    # 21: dynamic
        [55, 90, 80],      # 22: other
        [45, 60, 150],     # 23: water
        [157, 234, 50],    # 24: road line
        [81, 0, 81],       # 25: ground
        [150, 100, 100],   # 26: bridge
        [230, 150, 140],   # 27: rail track
        [180, 165, 180]    # 28: guard rail
    ], dtype=np.uint8)

def masks_to_colorimg(masks):
    """
    Converts a stack of binary masks to a colored image using the Cityscapes palette.
    """
    colors = get_cityscapes_palette()

    # The model outputs NUM_CLASSES channels. Ensure the color palette is large enough.
    if masks.shape[0] > len(colors):
        raise ValueError(f"The model has {masks.shape[0]} output classes, but the color palette only has {len(colors)} colors.")

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    # Get the index of the class with the highest probability for each pixel
    pred_labels = np.argmax(masks, axis=0)

    for y in range(height):
        for x in range(width):
            # Check if any class has a probability > 0.5
            if np.max(masks[:, y, x]) > 0.5:
                class_index = pred_labels[y, x]
                colorimg[y, x, :] = colors[class_index]
            # Optionally, you can keep the background white or set it to the 'unlabeled' color
            else:
                colorimg[y, x, :] = colors[0] # Black for unlabeled

    return colorimg.astype(np.uint8)