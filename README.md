# Fisheye_correction
参考代码:https://github.com/HLearning/fisheye  
# 原图像
![](https://github.com/smallboxx/Fisheye_correction/blob/main/imgs/test3.jpg)
# 校正效果
<div style="display: flex;">
    <img src="https://github.com/smallboxx/Fisheye_correction/blob/main/undistorted_by_R161.jpg" style="width:50%;" />
    <img src="https://github.com/smallboxx/Fisheye_correction/blob/main/undistorted_by_R321.jpg" style="width:50%;" />
</div>
# 代码说明
correction.py中进行鱼眼双经度校正，参考代码中坐标逆映射采用正交投影方法，本代码更换为等距投影方法，并且参考代码中求R的方式与《基于双经度模型的鱼眼图像畸变校正方法》不一致，会对结果产生一定影响，为了方便测试，采用了基于鱼眼镜头的R范围校正，以最终确定效果较好的R
