from PIL import Image
from pylab import *
from collections import defaultdict
import cv2
import os
import shutil
import tensorflow as tf
import pickle
from sklearn.cluster import DBSCAN
import numpy as np
import time


def process_img(fileName, sess):
    blocks = []
    im = np.array(cv2.imread(fileName))[:,:,[2,1,0]]
    x, y, channels = im.shape
    if y>x:
        im = np.rot90(im, -1)
        x, y, channels = im.shape
    if x*y>1000000:
        new_x, new_y = int(np.sqrt(x*1000000/y)), int(np.sqrt(y*1000000/x))
    else:
        new_x = x
        new_y = y
    out = np.array(Image.fromarray(im).resize((new_y,new_x)))
    im = np.array(Image.fromarray(im).resize((new_y,new_x)))
    r = im[:,:,0]/255
    g = im[:,:,1]/255
    b = im[:,:,2]/255
    T = np.zeros(r.shape)
    T = r - g + (b - g) * 6 + (r + g + b) / 3
    T = (T - np.min(T))/(np.max(T) - np.min(T))
    T = T>np.sum(T)/(T.shape[0]*T.shape[1])
    rho = []

    for i in range(new_x//150):
        for j in range(new_y//150):
            block = out[i*150:(i+1)*150, j*150:(j+1)*150]
            blocks.append(block)
            rho.append(np.sum(T[i*150:(i+1)*150, j*150:(j+1)*150])/(150*150))
    rho = np.reshape(rho, (new_x//150, new_y//150))
    rho = rho>0.12

    input = sess.graph.get_tensor_by_name("x:0")
    output = sess.graph.get_tensor_by_name("dnn/outputs/results:0")

    results = sess.run(output, feed_dict={input:blocks})
    results = np.exp(results)[:,1]/np.sum(np.exp(results), axis=1)
    results = results.reshape((new_x//150, new_y//150))
    results = rho*results
    cross_sum_mat = []
    for i in range(1,new_x//150-1):
        for j in range(1,new_y//150-1):
            cross_sum = (results[i,j]+results[i-1,j]+results[i+1,j]+results[i,j-1]+results[i,j+1])/5
            cross_sum_mat.append(cross_sum)
    cross_sum_mat = np.array(cross_sum_mat)
    cross_sum_mat = cross_sum_mat.reshape((new_x//150-2,new_y//150-2))
    central_x = np.argmax(cross_sum_mat)//(new_y//150-2)+1
    central_y = np.argmax(cross_sum_mat)%(new_y//150-2)+1
    top, bottom, left, right = central_x, central_x, central_y, central_y
    while True:
        top = top - 1
        if results[top, central_y]<0.8:
            break
        if top == 0:
            break
    while True:
        bottom = bottom + 1
        if results[bottom, central_y]<0.8 and rho[bottom, central_y]<rho[bottom-1, central_y]:
            break
        if bottom == new_x//150-1:
            break
    while True:
        left = left - 1
        if results[central_x, left]<0.8 and rho[central_x, left]<rho[central_x, left+1]:
            break
        if left == 0:
            break
    while True:
        right = right + 1
        if results[central_x, right]<0.8 and rho[central_x, right]<rho[central_x, right-1]:
            break
        if right == new_y//150-1:
            break

    if top-central_x==-1:
        top_ = max(central_x*150-75-int(150*results[top, central_y]),central_x*150-150)
    else:
        top_ = (top+1)*150-int(150*results[top, central_y])
    if bottom-central_x==1:
        bottom_ = min((central_x+1)*150+75+int(150*results[bottom, central_y]),(central_x+1)*150+150)
    else:
        bottom_ = bottom*150+int(150*results[bottom, central_y])
    if left-central_y==-1:
        left_ = max(central_y*150-75-int(150*results[central_x, left]),central_y*150-150)
    else:
        left_ = (left+1)*150-int(150*results[central_x, left])
    if right-central_y==1:
        right_ = min((central_y+1)*150+75+int(150*results[central_x, right]),(central_y+1)*150+150)
    else:
        right_ = right*150+int(150*results[central_x, right])

    stride = 64
    orig = out
    mosaic = im
    for i in range(new_x//stride+1):
        for j in range(new_y//stride+1):
            mosaic[i*stride:(i+1)*stride,j*stride:(j+1)*stride,:]=mosaic[i*stride,j*stride,:]
    mosaic[top_:bottom_,left_:right_] = orig[top_:bottom_,left_:right_]
    # mosaic = Image.fromarray(mosaic)
    # mosaic.save('temp3.jpg')

    im = out[top_:bottom_, left_:right_]
    return [mosaic, im]


def get_tongue(im_ori):
    red = im_ori[:,:,2]
    green = im_ori[:,:,1]
    blue = im_ori[:,:,0]
    r = np.float64(red)
    g = np.float64(green)
    b = np.float64(blue)
    T = np.zeros(r.shape)
    T = r - g + (b - g)*6 + (r+g+b)/3
    T = T>np.sum(T)/(T.shape[0]*T.shape[1])
    T = np.float64(T)
    im = np.array(Image.fromarray(T).resize((100,100)))
    points = np.array(np.where(im==1)).T
    label_pred = DBSCAN(eps=2).fit_predict(points)
    cluster_dict = defaultdict(list)
    for label, point in zip(label_pred, points):
        cluster_dict[label].append(point)
    clusters = sorted(cluster_dict.values(), key=lambda x:len(x),reverse=True)
    clusters = clusters[0]
    col_dicts = defaultdict(list)
    row_dicts = defaultdict(list)
    for point in clusters:
        row_dicts[point[0]].append(point[1])
        col_dicts[point[1]].append(point[0])
    con_hull = np.zeros((100,100))
    for key in row_dicts.keys():
        con_hull[key,min(row_dicts[key]): max(row_dicts[key])] = 1
    for key in col_dicts.keys():
        con_hull[min(col_dicts[key]): max(col_dicts[key]), key] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    con_hull = cv2.erode(con_hull, kernel)
    con_hull = cv2.dilate(con_hull, kernel)
    im = np.array(Image.fromarray(con_hull).resize((T.shape[1],T.shape[0])))
    tongue = np.uint8(np.zeros((im_ori.shape)))
    tongue[np.where(im==1)] = im_ori[np.where(im==1)]
    return tongue


def split(im):
    im_l = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)[:,:,0]
    l_thr = max(int(np.sum(im_l)/np.sum(im_l>0) + 65), 250)
    im[np.where(im_l>l_thr)] = 0
    im_gray = im[:,:,1]
    start = int(np.sum(im_gray)/np.sum(im_gray>0))
    threshold = np.arange(start,255)
    thres = 0
    g_max = 0
    for item in threshold:
        tmp1 = im_gray>start-1
        tmp2 = im_gray<item
        points1 = np.where(tmp1&tmp2)
        points2 = np.where(im_gray>=item)
        w1 = len(points1[0])/(len(points1[0])+len(points2[0]))
        w2 = 1 - w1
        mu1 = np.sum(im_gray[points1])/len(points1[0])
        mu2 = np.sum(im_gray[points2])/len(points2[0])
        g = w1*w2*(mu1-mu2)**2
        if g>g_max:
            g_max = g
            thres = item
    tmp = im_gray>=thres
    r = np.float64(im[:,:,0])
    g = np.float64(im[:,:,1])
    b = np.float64(im[:,:,2])
    T = np.zeros(r.shape)
    T = r - g + (b - g)*6 + (r+g+b)/3
    T = T<=np.sum(T)/(np.sum(T>0))
    T = T & (im_l>50)
    shetai = T | tmp
    shezhi = ~shetai & (im_l>50)

    return (shetai, shezhi)


def analyze_shezhi(shezhi, im):
#     im = cv2.imread(file_path)
    b_data = im[:,:,0][np.where(shezhi)]
    g_data = im[:,:,1][np.where(shezhi)]
    r_data = im[:,:,2][np.where(shezhi)]

    r_mean = np.mean(r_data)
    g_mean = np.mean(g_data)
    b_mean = np.mean(b_data)

    if r_mean <= 170:
        return ('舌质淡白')
    elif r_mean <= 210:
        return('舌质淡红')
    else:
        if b_mean < 125: ###一开始我好像弄错了，写了大于号
            return('舌质红')
        else:
            return('舌质绛')


def analyze_shetai(shetai, im):
#     im = cv2.imread(file_path)
    b_data = im[:,:,0][np.where(shetai)]
    g_data = im[:,:,1][np.where(shetai)]
    r_data = im[:,:,2][np.where(shetai)]

    r_mean = np.mean(r_data)
    g_mean = np.mean(g_data)
    b_mean = np.mean(b_data)

#     if b_mean > 140:
#         print('舌苔白')
#     else:
#         print('舌苔黄')
    if g_mean - b_mean > 18:
        print(g_mean-b_mean)
        return('舌苔黄')
    else:
        return('舌苔白')


def workflow(fileName):
    im = process_img(fileName)
    im_lab = cv2.cvtColor(im[:,:,[2,1,0]], cv2.COLOR_BGR2Lab)
    l = np.sum(im_lab[:,:,0])/(im_lab[:,:,0].shape[0]*im_lab[:,:,0].shape[1])
    im_lab = np.float64(im_lab)
    im_lab[:,:,0] += 150 - l
    im_lab[im_lab>255]=255
    im_lab[im_lab<0]=0
    im_lab = np.uint8(im_lab)
    im = cv2.cvtColor(im_lab,cv2.COLOR_Lab2BGR)

    img = Image.fromarray(im[:,:,[2,1,0]])
    img.save('tmp.jpg','jpeg')

    tongue = get_tongue(im)
    im_lab = cv2.cvtColor(tongue, cv2.COLOR_BGR2Lab)
    im_l = im_lab[:,:,0]
    l = np.sum(im_l)/np.sum(im_l>0)
    im_l = np.float64(im_l)
    im_l[np.where(im_l>0)] += 150 - l
    im_l[im_l>255]=255
    im_l[im_l<0]=0
    im_l = np.uint8(im_l)
    im_lab[:,:,0] = im_l
    tongue = cv2.cvtColor(im_lab,cv2.COLOR_Lab2BGR)
    shetai, shezhi = split(tongue)
    print(analyze_shezhi(shezhi, tongue))
    if np.sum(shetai)/np.sum(shezhi)>0.5:
        print(analyze_shetai(shetai, tongue))


def setup(fileName):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = fileName
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        sess = tf.Session()
        return sess


def analyze(fileName, sess):
    mosaic, im = process_img(fileName, sess)
    im_lab = cv2.cvtColor(im[:,:,[2,1,0]], cv2.COLOR_BGR2Lab)
    l = np.sum(im_lab[:,:,0])/(im_lab[:,:,0].shape[0]*im_lab[:,:,0].shape[1])
    im_lab = np.float64(im_lab)
    im_lab[:,:,0] += 150 - l
    im_lab[im_lab>255]=255
    im_lab[im_lab<0]=0
    im_lab = np.uint8(im_lab)
    im = cv2.cvtColor(im_lab,cv2.COLOR_Lab2BGR)

    tongue = get_tongue(im)
    im_lab = cv2.cvtColor(tongue, cv2.COLOR_BGR2Lab)
    im_l = im_lab[:,:,0]
    l = np.sum(im_l)/np.sum(im_l>0)
    im_l = np.float64(im_l)
    im_l[np.where(im_l>0)] += 150 - l
    im_l[im_l>255]=255
    im_l[im_l<0]=0
    im_l = np.uint8(im_l)
    im_lab[:,:,0] = im_l
    tongue = cv2.cvtColor(im_lab,cv2.COLOR_Lab2BGR)
    shetai, shezhi = split(tongue)
    shezhi_res = analyze_shezhi(shezhi, tongue)
    shetai_res = ''
    if np.sum(shetai)/np.sum(shezhi)>0.5:
        shetai_res = analyze_shetai(shetai, tongue)

    return {'mosaic_img':mosaic, 'tongue_img':im[:,:,[2,1,0]], 'shezhi':shezhi_res, 'shetai':shetai_res}
