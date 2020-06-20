import numpy as np
from skimage.feature import corner_shi_tomasi, corner_peaks
import matplotlib.pyplot as plt

def getFeatures(img, bbox):
    x = np.empty((0, 0))
    y = np.empty((0, 0))

    for idx, box in enumerate(bbox):
        box = box.astype(int)
        print(box)
        boxTopLeftX = np.min(box[:, 0])
        boxTopLeftY = np.min(box[:, 1])
        boxW = np.max(box[:, 0]) - boxTopLeftX
        boxH = np.max(box[:, 1]) - boxTopLeftY
        extracted_box = img[boxTopLeftY:boxTopLeftY + boxH, boxTopLeftX:boxTopLeftX + boxW]
        # plt.figure(1)
        # plt.imshow(extracted_box)
        # plt.show()
        pts = corner_peaks(corner_shi_tomasi(extracted_box),min_distance=1, num_peaks=50)
        x_idx = np.reshape(pts[:, 0]+boxTopLeftY, (-1, 1))
        y_idx = np.reshape(pts[:, 1]+boxTopLeftX, (-1, 1))

        if(x.shape[0] == 0):
            x = np.resize(x, (x_idx.shape[0], x.shape[1]))
            y = np.resize(y, (y_idx.shape[0], y.shape[1]))
            x = np.append(x, x_idx, axis=1)
            y = np.append(y, y_idx, axis=1)
        elif(x.shape[0] < x_idx.shape[0]):
            N = x_idx.shape[0]
            x = np.pad(x, ((0, N-x.shape[0]), (0, 0)),'constant', constant_values=(-1))
            y = np.pad(y, ((0, N-y.shape[0]), (0, 0)),'constant', constant_values=(-1))
            x = np.append(x, x_idx, axis=1)
            y = np.append(y, y_idx, axis=1)
        else:
            N = x.shape[0]
            x_idx = np.pad(x_idx, ((0, N-x_idx.shape[0]), (0, 0)), 'constant', constant_values=(-1))
            y_idx = np.pad(y_idx, ((0, N-y_idx.shape[0]), (0, 0)),'constant', constant_values=(-1))
            x = np.append(x, x_idx, axis=1)
            y = np.append(y, y_idx, axis=1)
    return x, y
