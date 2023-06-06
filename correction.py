import cv2
import numpy as np
import glob
import time
import os

# 鱼眼有效区域截取
def int_cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/ 2, h/ 2)
    # 提取有效区域
    img_valid = img[y:y+h, x:x+w]
    return img_valid, int(r)

# 鱼眼矫正
def int_undistort(src,r):
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
    u[u>(src_w-1)] = src_w-1
    v[v<0]=0
    v[v>(src_h-1)] = src_h-1

    # 插值
    dst[:, :, :] = src[v.astype(int),u.astype(int)]
    return dst
 
def get_K_and_D(checkerboard, imgsPath):
 
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []
    imgpoints = []
    images = os.listdir(imgsPath)
    for fname in images:
        name = imgsPath+"/"+fname
        print(name)
        img = cv2.imread(name)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(ret,corners)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    print(objpoints,imgpoints)
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    return DIM, K, D
 
 
def undistort(img,K,D,DIM,scale=0.6,imshow=False):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0]!=DIM[0]:
        img = cv2.resize(img,DIM,interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:#change fov
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img


if __name__ == '__main__':
 
   # 开始使用图片来获取内参和畸变系数
   # DIM, K, D = get_K_and_D((6,9), 'C:/Users/alienware/Documents/GitHub/Fisheye_correction/imgs')
 
   # 得到内参和畸变系数畸变矫正进行测试
    '''
    DIM=(2560, 1920)
    K=np.array([[652.8609862494474, 0.0, 1262.1021584894233], [0.0, 653.1909758659955, 928.0871455436396], [0.0, 0.0, 1.0]])
    D=np.array([[-0.024092199861108887], [0.002745976275100771], [0.002545415522352827], [-0.0014366825722748522]])
    img = undistort('../imgs/pig.jpg',K,D,DIM)
    cv2.imwrite('../imgs/pig_checkerboard.jpg', img)
    '''
    DIM=(640, 480)
    K=np.array([[246.22820045943067, 0.0, 343.7456550187162], [0.0, 246.1938240062321, 197.31886375433282], [0.0, 0.0, 1.0]])
    D=np.array([[-0.036824195555787205], [0.027762257179681806], [-0.026728375721557774], [0.0069573833763686916]])
    PATH = 'cam'
    
    if PATH == 'cam':
        cap = cv2.VideoCapture(2)
        while(True):
            ret,frame=cap.read()
            if ret:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                cut_img,R = int_cut(frame)
                undistorted_img = int_undistort(cut_img,R)
                img = undistort(frame,K,D,DIM)
                undistorted_img = cv2.convertScaleAbs(undistorted_img)
                cv2.imshow("frame",frame)
                cv2.imshow('undistort', img)
                cv2.imshow('undistortImage',undistorted_img)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread('imgs/img113.jpg')
        img = undistort(frame,K,D,DIM)
        cv2.imwrite('imgs/pig_checkerboard.jpg', img)