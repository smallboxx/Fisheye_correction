import cv2
import math
import numpy as np

# 鱼眼有效区域截取
def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #(_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/ 2, h/ 2)
    # 提取有效区域
    img_valid = img[y:y+h, x:x+w]
    print(x,y,w,h,r)
    # x=147
    # y=0
    # w=503
    # h=432
    # r=251.0
    # img_valid = img[y:y+h, x:x+w]
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
    filename="undistorted_by_R"+str(r)+".jpg"
    cv2.imwrite(filename,dst)
    print('complete!')
    return dst

if __name__ == "__main__":
    img = cv2.imread("imgs\\test3.jpg")
    print(img.shape)
    img = cv2.resize(img,(768,432)) 
    cut_img,r = cut(img)
    r_list=[]
    for i in range(10):
        if i ==0 :
            r_list.append(r)
            continue
        r_list.append(r-i*10)
        r_list.append(r+i*10)
    print(r_list)
    for r_index in r_list: 
        output = my_undistorted(cut_img,r_index)