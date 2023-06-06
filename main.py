import cv2
import numpy as np
import math

# 鱼眼有效区域截取
def cut(img):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # #(_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    #                            cv2.THRESH_BINARY, 11, 2)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # x,y,w,h = cv2.boundingRect(cnts)
    # r = max(w/ 2, h/ 2)
    # # 提取有效区域
    # img_valid = img[y:y+h, x:x+w]

    # circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
    #                        param1=50, param2=30, minRadius=200, maxRadius=0)

    # if circles is not None:
    #     # 将检测到的圆形物体画出来
    #     circles = np.uint16(np.around(circles))
    #     rad = circles[0,:,2]
    #     idx = np.argsort(rad)
    #     sort_circles = circles[0,idx]
    #     x,y,r=sort_circles[0]
    #     cv2.circle(img, (x,y), r, (0, 255, 0), 2)

    # cv2.imshow('image', img)
    x=65
    y=0
    w=540
    h=480
    r=270
    img_valid = img[y:y+h, x:x+w]
    return img_valid, int(r)

# 鱼眼矫正
def undistort(src,r):
    # r： 半径， R: 直径
    R = 2*r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape

    # 圆心
    x0, y0 = src_w//2, src_h//2

    # 数组， 循环每个点
    range_arr = np.array([range(R)])

    theta = Pi - (Pi/R)*(range_arr.T)
    temp_theta = np.tan(theta)**2

    phi = Pi - (Pi/R)*range_arr
    temp_phi = np.tan(phi)**2

    tempu = r/(temp_phi + 1 + temp_phi/temp_theta)**0.5
    tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

    # 用于修正正负号
    flag = np.array([-1] * r + [1] * r)

    # 加0.5是为了四舍五入求最近点
    u = x0 + tempu * flag + 0.5
    v = y0 + tempv * np.array([flag]).T + 0.5

    # 防止数组溢出
    u[u<0]=0
    u[u>(src_w-1)] = 0
    v[v<0]=0
    v[v>(src_h-1)] = 0
    # 插值
    dst[:, :, :] = src[v.astype(int),u.astype(int)]
    return dst

def my_undistorted(src,r):
    # r： 半径， R: 直径
    D = 2*r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((D, D, 3))
    src_h, src_w, _ = src.shape
    # 圆心
    x0, y0 = src_w//2, src_h//2

    for dst_y in range(0, D):

        theta =  Pi - (Pi/D)*dst_y
        temp_theta = pow(math.tan(theta),2)

        for dst_x in range(0, D):
            # 取坐标点 p[i][j]
            phi = Pi - (Pi/D)*dst_x
            temp_phi = pow(math.tan(phi),2)

            x = r/math.sqrt(temp_phi+ 1 + temp_phi/temp_theta)
            y = r/math.sqrt(temp_theta + 1 + temp_theta/temp_phi)
            z = r/math.sqrt(1+1/temp_phi+1/temp_theta)

            if (phi < Pi/2):
                u = x0 + r*np.arccos(z/r)*(x/math.sqrt(x**2+y**2))
            else:
                u = x0 - r*np.arccos(z/r)*(x/math.sqrt(x**2+y**2))

            if (theta < Pi/2):
                v = y0 + r*np.arccos(z/r)*(y/math.sqrt(x**2+y**2))
            else:
                v = y0 - r*np.arccos(z/r)*(y/math.sqrt(x**2+y**2))

            if (u>=0 and v>=0 and u+0.5<src_w and v+0.5<src_h):
                # 计算在源图上四个近邻点的位置
                src_x, src_y = u, v
                src_x_0 = int(src_x)
                src_y_0 = int(src_y)
                src_x_1 = min(src_x_0 + 1, src_w - 1)
                src_y_1 = min(src_y_0 + 1, src_h - 1)
                
                value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, :] + (src_x - src_x_0) * src[src_y_0, src_x_1, :]
                value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, :] + (src_x - src_x_0) * src[src_y_1, src_x_1, :]
                dst[dst_y, dst_x, :] = ((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1 + 0.5).astype('uint8')
    cv2.imwrite("org.jpg",src)
    cv2.imwrite("mask.jpg",dst)
    print('coo')
    return dst

if __name__ == "__main__":

    # frame = cv2.imread('imgs/img113.jpg')
    # cv2.imshow("2",frame)
    # cut_img,R = cut(frame)
    # result_img = undistort(cut_img,R)
    # img = cv2.convertScaleAbs(result_img)
    # print(img)
    # cv2.imshow("d",img)
    # cv2.waitKey(0)
    # cv2.imwrite('imgs/pig_vector_nearest.jpg',result_img)
    cam = cv2.VideoCapture(2)
    while(True):
        ret,frame = cam.read()
        if ret:
            cv2.imshow("frame",frame)
            cut_img,r = cut(frame)
            # result_img = undistort(cut_img,R)
            # undistort_img = cv2.convertScaleAbs(result_img)
            # cv2.imshow("da",cut_img)
            # cv2.imshow("undistort_img",undistort_img)

            dst = my_undistorted(cut_img,r)

        if cv2.waitKey(1) == ord('q'):
            break