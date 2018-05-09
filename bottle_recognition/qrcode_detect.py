import numpy as np
import cv2
import os
import shutil

def resize(img):
    '''
    等比缩放图片，保证图片尺寸在合理范围内

    Args：
    `img`-> 传入的图像，ndarray型
    '''
	# 等比缩放图片
    h, w = img.shape[:2]
    if max(w, h) > 1024:
        scale = max(w, h) / 1024
        w_scaled = w // scale
        h_scaled = h // scale
        img = cv2.resize(img, (int(w_scaled), int(h_scaled)),
                        interpolation = cv2.INTER_AREA)
    return img.copy()
# img_gray = cv2.resize(img_gray, (int(w_scaled), int(h_scaled)),
#                 interpolation=cv2.INTER_AREA)

def qrcode_detect(img, img_gray, flag=True):
    '''
    检测图片中的二维码区域

    Args:
    `img` -> 传入的图像，ndarray型
    `img_gray` -> 传入图像的灰度图，ndarray型
    `flag` -> 返回数据选择标识

    Return:
    flag为`true`时返回二维码定位点(左上角和右下角), [min_point, max_point]
    flag为`false`时返回图片的倾斜度，int型
    '''
    # cv2.imshow("test", image)
    # 简单滤波
    # ret1, th1=cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bin", th1)
    # Otsu 滤波, 得到二值图
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + \
                                cv2.THRESH_OTSU)
    # cv2.imshow("otsu", img_bin)
    # 寻找轮廓
    # cv2.RETR_LIST检测的轮廓不建立等级关系
    #  cv2.CHAIN_APPROX_SIMPLE压缩水平方向，
    # 垂直方向，对角线方向的元素，只保留该方向的终点坐标，
    # 例如一个矩形轮廓只需4个点来保存轮廓信息
    '''hierarchy :[Next. Previous, First Child, Parten]
    Next: 同一级的下一个框，-1表示没有
    Previous：同一级的上一个框，-1表示没有
    First child: 第一个子轮廓
    Parent：父轮廓
    '''
    '''findcontours模式：
    CV_RETR_EXTERNAL: 只检索外部轮廓，没有层级关系
    CV_RETR_LIST：检索所有轮廓，但都当作同一级处理，没有父子关系
    CV_RETR_CCOMP: 检索所有轮廓，按照二级结构组织（外轮廓和内轮廓）
    CV_RETR_TREE: 检索所有轮廓，按照树形目录结构组织
    '''
    _, contours, hierarchys = cv2.findContours(img_bin, cv2.RETR_TREE,\
                                            cv2.CHAIN_APPROX_SIMPLE)
    # 在图像上绘制轮廓
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # 显示绘制所有的轮廓
    # cv2.imshow("add contours", img)
    # 先通过嵌套关系，找到带选择的轮廓
    # 即找到有2个以上内轮廓的轮廓
    hierarchys = hierarchys[0] # findcontours返回的是一个三维数组
    index_filter1 = []
    for i in range(len(hierarchys)):
        k = i
        c = 0
        while hierarchys[k][2] != -1:# 说明有内轮廓
            k = hierarchys[k][2]
            c += 1
        if c == 2:
            index_filter1.append(i)
    # 显示找到的待选外框
    # for i in index_filter1:
    #     cv2.drawContours(img, contours, i, (0, 0, 255), 3)
    #     cv2.imshow("add contours", img)
    #     cv2.waitKey(2000)

    # 用其他特征进一步筛选
    max_points_x = []
    max_points_y = []
    min_points_x = []
    min_points_y = []
    angle_r = []
    for i in index_filter1:
        contour = contours[i]
        # 求出满足条件轮廓的最小包围矩形
        # 返回((中心点坐标x, y)，(长, 宽)，旋转角度[-90,0))，
        # 旋转角度θ是水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边的夹角。
        rect = cv2.minAreaRect(contour)
        rate_wh = min(rect[1]) / max(rect[1]) #长宽比
        # 筛选长宽比高于0.85并且长宽小于原图1/4的轮廓
        if rate_wh > 0.85 and rect[1][0] < len(img[0]) / 4 \
                        and rect[1][0] < len(img) / 4:
            # 判断内外框长度比是否在7:5:3左右
            # 找到此框的两个内框，并求minrect
            index_child = hierarchys[i][2]
            contour_child = contours[index_child]
            index_grandchild = hierarchys[index_child][2]
            contour_grandchild = contours[index_grandchild]
            rect_child = cv2.minAreaRect(contour_child)
            rect_grandchild = cv2.minAreaRect(contour_grandchild)
            # 判断倾斜度是否一致
            parent_r = 0 if rect[2] == -90 else rect[2]
            child_r = 0 if rect_child[2] == -90 else rect_child[2]
            grandchild_r = 0 if rect_grandchild[2] == -90 else rect_grandchild[2]
            std_dev_r = np.std([parent_r, child_r, grandchild_r])
            if std_dev_r > 5:
                continue
            else:
                angle_r.append((parent_r+child_r+grandchild_r)/3)
            # 判断圆心是否重合，三个圆心点在x， y的标准差
            std_dev_x = np.std([rect[0][0], rect_child[0][0], rect_grandchild[0][0]])
            std_dev_y = np.std([rect[0][1], rect_child[0][1], rect_grandchild[0][1]])
            if std_dev_x > 50 or std_dev_y > 50:
                continue
            # 判断x轴上长度比
            w_parent = rect[1][0]
            w_child = rect_child[1][0]
            w_grandchild = rect_grandchild[1][0]
            child_rate_x = w_child / w_parent
            grandchild_rate_x = w_grandchild / w_child
            if child_rate_x > 0.8 or child_rate_x < 0.6:
                continue
            if grandchild_rate_x > 0.7 or grandchild_rate_x < 0.5:
                continue
            # 判断y轴上宽度比
            h_parent = rect[1][1]
            h_child = rect_child[1][1]
            h_grandchild = rect_grandchild[1][1]
            child_rate_y = h_child / h_parent
            grandchild_rate_y = h_grandchild / h_child
            if child_rate_y > 0.8 or child_rate_y < 0.6:
                continue
            if grandchild_rate_y > 0.7 or grandchild_rate_y < 0.5:
                continue
            # 选出x，y最小和最大的点
            # print(contour)
            # print("==============")
            max_points_x.append(np.max(contour, 0).tolist()[0][0])
            min_points_x.append(np.min(contour, 0).tolist()[0][0])
            max_points_y.append(np.max(contour, 0).tolist()[0][1])
            min_points_y.append(np.min(contour, 0).tolist()[0][1])
            # 判断完成，画出所有框
            # cv2.drawContours(img, contours, i, (0, 0, 255), 3)
            # cv2.imshow("add contours", img)
            # cv2.waitKey(500)
    # 找到最外围框的两个定位点
    if not max_points_x:
        return False
    max_point = (max(max_points_x), max(max_points_y))
    min_point = (min(min_points_x), min(min_points_y))
    # 需要返回二维码定位点信息
    if flag is True:
        # cv2.rectangle(img, min_point, max_point, (0, 0, 255), 3)
        # cv2.imshow("add contours", img)
        # cv2.waitKey(500)
        # cv2.destroyWindow("add contours")
        return min_point, max_point

    # 只需要倾斜度
    if flag is False:
        angle_rotate = sum(angle_r) / len(angle_r)
        return angle_rotate


# 定义旋转函数
def rotate(image, angle, center=None, scale=0.8):
    '''
    以图片中心点为原点旋转图片

    Args:
    `image` -> 输入图片，ndarray型
    `angle` -> 旋转角度，int型
    `center` -> 旋转中心，默认为图片中心，[x, y]型
    `scale` -> 缩放尺寸，为了保证调整后图像完整，默认0.8

    Return: 调整后的image，ndarray型
    '''
    # 获取图像尺寸
    (h, w) = image.shape[0:2]
    # 若未指定旋转中心，则将图像中心设为旋转点
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # 返回旋转后的图像
    return rotated

def image_handle(path):
    '''
    图片处理，输入图片地址，将会在本目录建立一个以图片名称命名的文件夹，
    里面包含有0.jpg, 1.jpg和2.jpg三张分割好的图片

    Args:
    `path` -> 图片地址，string型

    Return: `True` or `False`, 处理是否成功
    '''
    # 读入图片
    img = cv2.imread(path)
    # resize 图片
    img_resized = resize(img)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    angle = qrcode_detect(img_resized, img_gray, False)
    if angle is False:
        return False
    if angle < -45:
        angle = 90 + angle
    img = resize(img)
    if angle != 0:
        img_rotated = rotate(img, angle)
        # 保存旋转后的图
        # cv2.imwrite("rotated.jpg", img_rotated)
        # cv2.imshow("rotated", img_rotated)
        # cv2.waitKey(500)
        # cv2.destroyWindow("rotated")
    else:
        img_rotated = img
    img_gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
    # (x_min, y_min), (x_max, y_max) 是二维码的两个定位点
    (x_min, y_min), (x_max, y_max) = qrcode_detect(img_rotated, img_gray)
    # 根据二维码位置得到总体位置
    qrcode_w = x_max - x_min
    qrcode_h = y_max - y_min
    # max这个点不变，只需算出大框的左上角位置(x_min_target, y_min_target)
    x_min_target = x_max - 4 * qrcode_w
    y_min_target = y_max - 2 * qrcode_h
    # cv2.rectangle(img_rotated, (x_max, y_max), (x_min_target, y_min_target),
    #                                 (0, 255, 0), 3)
    # cv2.imshow("find target", img_rotated)
    # cv2.waitKey(500)
    # 定位放瓶子的三个框
    target_w = x_max - x_min_target
    slide = target_w // 5
    # 计算出瓶子区域
    frame = []
    for i in range(3):
        point_lu = (x_min_target + i * slide, y_min_target) # 左上角的点
        point_rd = (x_min_target + (i+1) * slide, y_max) # 右下角的点
        frame.append([point_lu, point_rd])
        # cv2.rectangle(img_rotated, point_lu, point_rd, (0, 255, 0), 3)
        # cv2.imshow("find target", img_rotated)
        # cv2.waitKey(500)

    # 取出图片名称
    filename = os.path.split(path)[1].split('.')[0]

    # 如果文件夹已存在则删除
    if os.path.exists(filename):
        shutil.rmtree(filename)
    # 新建文件夹
    os.mkdir(filename)


    # 切分图片并保存
    flag = True
    for i, v in enumerate(frame):
        try:
            bottle = img_rotated[v[0][1]:v[1][1], v[0][0]:v[1][0]]
            path = filename + "\\"+ str(i) +".jpg"
            cv2.imwrite(path, bottle)
        except:
            flag = False

    return flag
