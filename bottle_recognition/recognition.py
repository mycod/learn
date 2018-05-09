import os
import shutil
from .qrcode_detect import image_handle
from .classify import image_classify

def recongnition(img_path):

    # 首先获得二维码信息
    cmd = "zbarimg -q " + img_path
    with os.popen(cmd) as code:
        info = code.read().lstrip('QR-Code:').strip('\n')

    bin_class, bin_id = info.split('_')
    
    # 调用二维码检测模块，切分待测图片
    # 如果切分失败，返回false
    slice_result = image_handle(img_path)
    if not slice_result:
        return False

    # 导入待检测图片
    # 取出图片名称
    filename = os.path.split(img_path)[1].split('.')[0]
    # 如果文件夹不存在则返回false
    if not os.path.exists(filename):
        return False
    
    classify_result = {'cans': 0, 'glass': 0, 'plastic': 0}
    for file in os.listdir(filename):
        file_path = filename + '\\' + file
        # 将图片分别送入分类网络
        label = image_classify(file_path)
        classify_result[label] += 1
    
    # 删除文件夹
    shutil.rmtree(filename)
    return [bin_id, bin_class, classify_result]

