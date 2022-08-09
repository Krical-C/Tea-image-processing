import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# 导入文件
os.chdir(r'E:\python_project\image-transform\image-example')
tif_list = [x for x in os.listdir() if x.endswith(".TIF")]
jpg_list = [x for x in os.listdir() if x.endswith(".JPG")]
print(tif_list)
print(jpg_list)


# 获取data和data的band
def get_data_band(path):
    img = gdal.Open(path)
    width = img.RasterXSize
    height = img.RasterYSize
    data = img.ReadAsArray(0, 0, width, height)
    data_max = data.max()
    data_min = data.min()
    # print(img,'\n',width,'\n',height,'\n',bands,'\n',geotrans,'\n',proj,'\n',data,'\n',data_max,'\n',data_min,'\n',end='\n')
    return data_min, data_max, data


def tiftoint8(min, max, imdata):
    scale = max - min
    out_img = ((imdata / scale) * 256).astype(np.uint8)
    return out_img


def init8totif(min, max, imdata):
    scale = max - min
    out_img = imdata.astype(np.float32) * scale / 256
    return out_img


def sift_kp(image):
    # gray_image=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)  #灰度图转换，如果是普通RGB图像需要这一行
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None)
    return kp_image, kp, des


def sift_kp_rgb(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图转换，如果是普通RGB图像需要这一行
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(gray_image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def surf_kp(image):
    '''SIFT(surf)特征点检测(速度比sift快)'''
    height, width = image.shape[:2]
    size = (int(width * 0.2), int(height * 0.2))
    shrink = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(shrink, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d_SURF.create()
    kp, des = surf.detectAndCompute(image, None)
    return kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    # matches=sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good


def siftImageAlignment(img1, img2):
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp_rgb(img2)
    goodMatch = get_good_match(des1, des2)
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return imgOut, H, status


step = 5

for i in range(0, len(tif_list), step):  # 遍历列表
    template_min, template_max, template_data = get_data_band(tif_list[i + 3])  # ir
    template_data_init8 = tiftoint8(template_min, template_max, template_data)
    print(tif_list[i + 3])
    print(jpg_list[i % 5])
    imgOut, _, _ = siftImageAlignment(template_data_init8, cv2.imread(jpg_list[i % 5]))
    cv2.imwrite(jpg_list[i % 5].split('.')[0] + "bgr.jpg", imgOut)
