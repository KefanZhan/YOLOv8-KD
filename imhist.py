import cv2
import matplotlib.pyplot as plt
#读图
path = ''
img = cv2.imread(path)
#转换成灰度图
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#显示灰度图
# cv2.imshow('gray_img',img2)
# cv2.waitKey(0)
#获取直方图，由于灰度图img2是二维数组，需转换成一维数组
plt.hist(img2.ravel(),256)
#显示直方图
plt.show()
# cv2.waitKey(0)

# 原图
# ori_t
# ori_s
# tar_t
# tar_s
# mask_tar_s