from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np


def get_tfb(img):
    i = img[:,:,1]
    j = img[:,:,0]
    b = img[:,:,2]
    f = cv.blur(i, ksize=(3,3)) + (cv.GaussianBlur(j,(3,3),4) - cv.GaussianBlur(j,(3,3),1))
    tfb = np.exp(-np.abs(f)/50)
#     return np.array([i,b,tfb]).transpose(1,2, 0)
    return tfb


def DIC_IOU(seg, gt):
    its = np.sum(seg[gt==255])
    seg_gt = (np.sum(seg) + np.sum(gt))
    un = seg_gt - its
    dice = its*2.0 / seg_gt * 100
    iou = its/un * 100
    
    return np.around(dice, decimals=2), np.around(iou, decimals=2)


def get_DIC_IOU(rgb_path, fg_path, init=np.array([[ 9.87324814], [27.68430863]])):
    prefix = './Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Tray'
    img = cv.imread(prefix + rgb_path)
    m,n,l = img.shape
    # bgr to lab
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    tfb = get_tfb(lab)
    X = tfb.reshape((-1,1))
    #from histogram thresholding in the Excess Green color space (Golzarian et al., 2012). 
    # Alternatively, in the presence of a plant appearance model we take advantage of it to get a good choice of initial cluster centroids 
#     kmeans = KMeans(n_clusters=2)
    kmeans = KMeans(n_clusters=2, init=init)
    y_hat = kmeans.fit_predict(X)
#     print(f'kmeans.cluster_centers_={kmeans.cluster_centers_}')
    y_hat = y_hat.astype(np.uint8)
    img = y_hat.reshape((m,n))
    img = img * 255
#     plt.axis('off')
#     plt.imshow(img, cmap='gray')
#     plt.show()
    gt = cv.imread(prefix + fg_path, 0)
    return DIC_IOU(img, gt)


def task2 ():
    res = []
    for idx in range(16):
        name = '{:02d}'.format(idx+1)
        rgb_path = f'/Ara2012/ara2012_tray{name}_rgb.png'
        fg_path = f'/Ara2012/ara2012_tray{name}_fg.png'
        res.append(list(get_DIC_IOU(rgb_path, fg_path, np.array([[ 9.87324814], [27.68430863]]))))

    for idx in range(27):
        name = '{:02d}'.format(idx+1)
        rgb_path = f'/Ara2013-Canon/ara2013_tray{name}_rgb.png'
        fg_path = f'/Ara2013-Canon/ara2013_tray{name}_fg.png'
        res.append(list(get_DIC_IOU(rgb_path, fg_path, np.array([[12.14169262], [23.3710466]]))))
#     print(f'Dice, IOU = {np.mean(res, axis=0)}')
    Dice, IOU = np.mean(res, axis=0)
    return Dice, IOU


task2()