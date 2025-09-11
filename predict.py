import cv2
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
sess = tf.InteractiveSession()
model_name = 'func/ffmpeg/modelData/semantic_model.meta'
# 加载权重
saver = tf.train.import_meta_graph(model_name)
saver.restore(sess, model_name[:-5])
graph = tf.get_default_graph()
input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

# 读取字典
dict_file = open('func/ffmpeg/modelData/vocabulary_semantic.txt', 'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()


def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]
    strs = [[] for i in range(dense_shape[0])]
    string = []
    ptr = 0
    b = 0
    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]
        string.append(values[ptr])
        ptr = ptr + 1
    strs[b] = string
    return strs


def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img


def normalize(image):
    return (255. - image) / 255.


def single_predict(img):
    notestxt = []
    img = resize(img, HEIGHT)
    img = normalize(img)
    img = np.asarray(img).reshape(1, img.shape[0], img.shape[1], 1)
    seq_lengths = [img.shape[2] / WIDTH_REDUCTION]
    prediction = sess.run(decoded,
                          feed_dict={
                              input: img,
                              seq_len: seq_lengths,
                              rnn_keep_prob: 1.0,
                          })
    str_prediction = sparse_tensor_to_strs(prediction)
    for i in str_prediction[0]:
        notestxt.append(int2word[i])
    return notestxt


def batch_predict(imgs):
    notestxt = []
    # 预测
    for img in imgs:
        img = resize(img, HEIGHT)
        img = normalize(img)
        img = np.asarray(img).reshape(1, img.shape[0], img.shape[1], 1)
        seq_lengths = [img.shape[2] / WIDTH_REDUCTION]
        prediction = sess.run(decoded,
                              feed_dict={
                                  input: img,
                                  seq_len: seq_lengths,
                                  rnn_keep_prob: 1.0,
                              })
        str_prediction = sparse_tensor_to_strs(prediction)
        for i in str_prediction[0]:
            notestxt.append(int2word[i])
    return notestxt
