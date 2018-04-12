import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from skimage.io import imread

from matplotlib.patches import Polygon
from skimage.measure import find_contours
import os
import inspect
import random
from PIL import Image

curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(curr_filename)

stage1_test_dir = root_dir + '/stage1_test'
test_id_list = sorted([test_id for test_id in os.listdir(stage1_test_dir) if not test_id.startswith('.')])


def get_test_image_by_id(img_id):
    img_path = os.path.join(stage1_test_dir, img_id, 'images', '%s.png' % img_id)
    return imread(img_path)[:, :, :3]


def get_random_color(pastel_factor=0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def rle_decode(rle_list, mask_shape, mask_dtype):

    # mask = np.zeros(mask_shape, dtype=mask_dtype)
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for j, rle in enumerate(rle_list):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = (j+1)

    return mask.reshape(mask_shape[::-1]).T


def get_polygon(mask, color_list=list()):
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    color = generate_new_color(color_list, pastel_factor=0.5)
    color_list.append(color)
    p_list = []
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        p_list.append(p)

    return p_list


def plot_mask_2d(img, mask_2d, bbox_list=None, class_probs=None, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(img)

    if title:
        ax.set_title(title)

    color_list = list()
    for j in range(1, np.max(mask_2d)+1):
        mask_j = mask_2d == j
        poly_list = get_polygon(mask=mask_j, color_list=color_list)
        for poly in poly_list:
            ax.add_patch(poly)

    if bbox_list is not None:
        for i_box in range(bbox_list.shape[0]):
            rect = patches.Rectangle((bbox_list[i_box][0], bbox_list[i_box][1]),
                                     bbox_list[i_box][2] - bbox_list[i_box][0],
                                     bbox_list[i_box][3] - bbox_list[i_box][1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            if class_probs is not None:
                ann_string = str(round(class_probs[i_box], 2))
                ax.annotate(ann_string, (bbox_list[i_box][0], bbox_list[i_box][1]), color='w',
                               weight='bold', fontsize=8, ha='left', va='bottom')


def get_test_mask(img_id):
    img_path = stage1_test_dir + '/' + img_id
    mask_files = next(os.walk(img_path + '/masks/'))[2]
    mask_files = sorted(mask_files)
    num_masks = len(mask_files)
    mask = imread(img_path + '/masks/' + mask_files[0])
    mask_2d = np.zeros(shape=[mask.shape[0], mask.shape[1]], dtype=np.int)

    for i_mask, mask_file in enumerate(mask_files):

        current_mask = imread(img_path + '/masks/' + mask_file)

        current_mask = (current_mask > 0).astype(int)

        assert np.sum(mask_2d[current_mask > 0]) == 0, 'Overlapping mask found!'

        mask_2d[current_mask > 0] = (i_mask + 1)

    return mask_2d


def sanity_check(max_n_images = 5):

    for i, test_id in enumerate(test_id_list):
        if i > max_n_images:
            break
        img = get_test_image_by_id(test_id)
        mask_2d = get_test_mask(img_id=test_id)
        plot_mask_2d(img=img, mask_2d=mask_2d)
        plt.show()


def decode_submission(filename, visualize=False):
    # filename = get_submission_path(run_name)

    df = pd.read_csv(filename, sep=',')

    for i, test_id in enumerate(test_id_list):

        print('Test ID = ', test_id)

        img = get_test_image_by_id(test_id)
        rows, cols = img.shape[0], img.shape[1]

        mask_rles = df.loc[df['ImageId'] == test_id]
        mask = rle_decode(rle_list=mask_rles['EncodedPixels'], mask_shape=(rows, cols), mask_dtype=np.int)

        if visualize:
            plot_mask_2d(img=img, mask_2d=mask)
            plt.show()

        num_masks = np.max(mask)

        mask_dir = stage1_test_dir + '/' + test_id + '/masks/'

        os.makedirs(mask_dir, exist_ok=True)

        for i_mask in range(num_masks):

            c_mask = 255 * (mask == (i_mask + 1))

            mask_img_string = test_id + '_' + str(i_mask) + '.png'

            im = Image.fromarray(np.asarray(c_mask, dtype=np.uint8))
            im.save(mask_dir + mask_img_string)


if __name__ == '__main__':
    # filename = 'stage1_solution.csv'
    # decode_submission(filename, visualize=False)

    sanity_check(max_n_images=15)