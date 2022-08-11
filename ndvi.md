图片：DJI_0150

### 1

```
# 消除 图像匹配后最右边边框出现的异常值
ndvi[ndvi > 0.75] = 0
# 去除出植物外的背景

```

##### NDVI：

![20220811103141](https://raw.githubusercontent.com/Krical-C/Tea-image-processing/master/record/20220811103141.png)

##### NDVI直方图：

![20220811103159](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811103159.png)



### 2

##### 第一次调整：

通过观察直方图两个波峰

用ostu算法找到阈值

```
# Otsu阈值
ret, th = cv2.threshold(np.uint8(ndvi*256), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret/256)
```

阈值为0.26171875

##### 选取0.261~0.750的范围：

ndvi：

![20220811104857](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811104857.png)

ndvi直方图：

![image-20220811105213596](https://github.com/Krical-C/Tea-image-processing/blob/master/record\image-20220811105213596.png)

![20220811105322](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811105322.png)

调整y值范围0~10000

![20220811105856](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811105856.png)

筛选图：

![DJI_0150bgr-0.261~0.750](https://github.com/Krical-C/Tea-image-processing/blob/master/record\DJI_0150bgr-0.261~0.750.jpg)

##### 选取0.550~0.750的范围：

ndvi：

![image-20220811113342647](https://github.com/Krical-C/Tea-image-processing/blob/master/record\image-20220811113342647.png)

ndvi直方图：

![20220811113444](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811113444.png)

筛选图：

![DJI_0150bgr-0.550~0.750](https://github.com/Krical-C/Tea-image-processing/blob/master/record\DJI_0150bgr-0.550~0.750.jpg)

##### 选取0.590~0.750的范围：

ndvi：

![20220811142931](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811142931.png)

nivi的直方图：

![20220811142951](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811142951.png)

筛选图：

![DJI_0150bgr-0.590~0.750](https://github.com/Krical-C/Tea-image-processing/blob/master/record\DJI_0150bgr-0.590~0.750.jpg)

##### 选取0.660~0.750的范围：

ndvi：

![20220811154456](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811154456.png)

nivi的直方图：

![20220811154540](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811154540.png)

筛选图：

![DJI_0150bgr-0.660~0.750](https://github.com/Krical-C/Tea-image-processing/blob/master/record\DJI_0150bgr-0.660~0.750.jpg)

##### 选取0.680~0.750的范围：

ndvi：

![20220811154826](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811154826.png)

nivi的直方图：

![20220811154859](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811154859.png)

筛选图：

![DJI_0150bgr-0.660~0.750](https://github.com/Krical-C/Tea-image-processing/blob/master/record\DJI_0150bgr-0.660~0.750.jpg)

##### 选取0.711~0.750的范围：

ndvi：

![20220811161402](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811161402.png)

nivi的直方图：

![20220811161431](https://github.com/Krical-C/Tea-image-processing/blob/master/record\20220811161431.png)

筛选图：

![DJI_0150bgr-0.711~0.750](https://github.com/Krical-C/Tea-image-processing/blob/master/record\DJI_0150bgr-0.711~0.750.jpg)
