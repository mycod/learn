import numpy as np
import tensorflow as tf

# 导入已经训练好的网络
def load_graph(model_file):
    '''
    导入已经训练好的神经网络

    Args: `model_file` -> tensorflow模型文件.pb

    Return: tensorflow中的graph类
    '''
    graph = tf.Graph()
    # 定义一个graph define，用来接收从string而来的网络连接
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        # 将文件中的网络模型转化为graphdef
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        # 将导入的模型设置为默认图
        tf.import_graph_def(graph_def)

    return graph

# 将要识别的图片转化为输入tensor
def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    '''
    将要识别的图片转化为输入tensor

    Args: 
    `file_name` -> 图片path，可以使jpg，png，bmp和gif
    `input_height`-> 所选用模型的input所需求的图片像素宽
    `input_width` -> 所选用模型的input所需求的图片像素高
    `input_mean` -> 
    `input_std`

    Return: tensorflow中的tensor类
    '''
    # 为input和output的tensor命名
    input_name = "file_reader"
    output_name = "normalized"
    # 读取所需要检测的图片
    file_reader = tf.read_file(file_name, input_name)
    # 对不同类型图片进行处理，转换为tensor
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels = 3, name = "jpeg_reader")
    # 将int类型的tensor转换为float32
    float_caster = tf.cast(image_reader, tf.float32)
    # 将图片tensor拉伸为一维
    dims_expander = tf.expand_dims(float_caster, 0)
    # 双线性插值resize到model所要求的尺寸
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    # 将像素值转换归一到[0, 1]之间
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    # 运行以上定义的操作，得到归一化的一维tensor
    result = sess.run(normalized)

    return result

# 导入label文件中的label
def load_labels(label_file):
    '''
    从labels.txt中读取label信息

    Args: `label_file` -> label文件的path

    Return: [label1, label2, ...]
    '''
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def image_classify(image_path):
    '''
    图片分类

    Args: `image_path` -> 需分类图片的path

    Return: 最大可能类的label，string类
    '''
    # 设置参数
    file_name = image_path
    model_file = "img_recognition/output_graph.pb"
    label_file = "img_recognition/output_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"

    # 导入model
    graph = load_graph(model_file)
    # 导入检测图片（tensor）
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)
    # 通过name找到调整后model中的input和output操作
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    # 开始计算
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)
    top_k = results.argsort()[::-1]
    labels = load_labels(label_file)
    return labels[top_k[0]]
