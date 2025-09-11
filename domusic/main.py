import os
import subprocess
import pickle
import glob
from music21 import converter, instrument, note, chord, stream  # converter负责转换,乐器，音符，和弦类
import tensorflow as tf
import numpy as np
from pydub import AudioSegment
import os
import transfer
import music21 as ms


# 构建神经网络模型
def network_model(inputs, num_pitch, weights_file=None):  # 输入，音符的数量，训练后的参数文件
    # 测试时要指定weights_file
    # 建立模子
    model = tf.keras.Sequential()

    # 第一层
    model.add(tf.keras.layers.LSTM(
        512,  # LSTM层神经元的数目是512，也是LSTM层输出的维度
        input_shape=(inputs.shape[1], inputs.shape[2]),  # 输入的形状，对于第一个LSTM必须设置
        return_sequences=True  # 返回控制类型，此时是返回所有的输出序列
        # True表示返回所有的输出序列
        # False表示返回输出序列的最后一个输出
        # 在堆叠的LSTM层时必须设置，最后一层LSTM不用设置，默认值为False
    ))

    # 第二层和第三层
    model.add(tf.keras.layers.Dropout(0.3))  # 丢弃30%神经元，防止过拟合
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))  # 丢弃30%神经元，防止过拟合
    model.add(tf.keras.layers.LSTM(512))  # 千万不要丢括号！！！！

    # 全连接层
    model.add(tf.keras.layers.Dense(256))  # 256个神经元的全连接层
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_pitch))  # 输出的数目等于所有不重复的音调数

    # 激活层
    model.add(tf.keras.layers.Activation('softmax'))  # Softmax激活函数求概率

    # 配置神经网络模型
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005))
    # 选择的损失函数是交叉熵，用来计算误差。使用对于RNN来说比较优秀的优化器-RMSProp
    # 优化器如果使用字符串的话会用默认参数导致效果不好

    if weights_file is not None:
        model.load_weights(weights_file)  # 就把这些参数加载到模型中，weight_file本身是HDF5文件
    return model


'''
def get_notes():
    """
    从music_midi目录中的所有MIDI文件里读取note，chord
    Note样例：B4，chord样例[C3,E4,G5],多个note的集合，统称“note”
    """
    notes = []
    for midi_file in glob.glob("melody/*.mid"):
        # 读取music_midi文件夹中所有的mid文件,file表示每一个文件
        stream = converter.parse(midi_file)  # midi文件的读取，解析，输出stream的流类型

        # 获取所有的乐器部分，开始测试的都是单轨的
        parts = instrument.partitionByInstrument(stream)
        if parts:  # 如果有乐器部分，取第一个乐器部分
            notes_to_parse = parts.parts[0].recurse()  # 递归
        else:
            notes_to_parse = stream.flat.notes  # 纯音符组成
        for element in notes_to_parse:  # notes本身不是字符串类型
            # 如果是note类型，取它的音高(pitch)
            if isinstance(element, note.Note):
                # 格式例如：E6
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # 转换后格式：45.21.78(midi_number)
                notes.append('.'.join(str(n) for n in element.normalOrder))  # 用.来分隔，把n按整数排序
    # 如果 data 目录不存在，创建此目录
    if not os.path.exists("data"):
        os.mkdir("data")
    # 将数据写入data/notes
    with open('data/notes', 'wb') as filepath:  # 从路径中打开文件，写入
        pickle.dump(notes, filepath)  # 把notes写入到文件中
    return notes  # 返回提取出来的notes列表
'''


def get_notes():
    filepath = './melody/'
    files = os.listdir(filepath)
    Notes = []
    for file in files:
        try:
            stream = converter.parse(filepath + file)
            print(stream)
            instru = instrument.partitionByInstrument(stream)
            print(instru)
            if instru:  # 如果有乐器部分，取第一个乐器部分
                notes = instru.parts[0].recurse()
                print(notes)
            else:  # 如果没有乐器部分，直接取note
                notes = stream.flat.notes
            for element in notes:
                # 如果是 Note 类型，取音调
                # 如果是 Chord 类型，取音调的序号,存int类型比较容易处理
                if isinstance(element, note.Note):
                    Notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    Notes.append('.'.join(str(n) for n in element.normalOrder))
        except:
            pass
        with open('func/ffmpeg/domusic//Notes', 'a+', encoding='utf-8') as f:
            f.write(str(Notes))
    return Notes


def create_music(prediction):  # 生成音乐函数，训练不用
    """ 用神经网络预测的音乐数据来生成mid文件 """
    offset = 0  # 偏移，防止数据覆盖
    output_notes = []
    # 生成Note或chord对象
    for data in prediction:
        # 如果是chord格式：45.21.78
        if ('.' in data) or data.isdigit():  # data中有.或者有数字
            note_in_chord = data.split('.')  # 用.分隔和弦中的每个音
            notes = []  # notes列表接收单音
            for current_note in note_in_chord:
                new_note = note.Note(int(current_note))  # 把当前音符化成整数，在对应midi_number转换成note
                new_note.storedInstrument = instrument.Piano()  # 乐器用钢琴
                notes.append(new_note)
            new_chord = chord.Chord(notes)  # 再把notes中的音化成新的和弦
            new_chord.offset = offset  # 初试定的偏移给和弦的偏移
            output_notes.append(new_chord)  # 把转化好的和弦传到output_notes中
        # 是note格式：
        else:
            new_note = note.Note(data)  # note直接可以把data变成新的note
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()  # 乐器用钢琴
            output_notes.append(new_note)  # 把new_note传到output_notes中
        # 每次迭代都将偏移增加，防止交叠覆盖
        offset += 0.5

    # 创建音乐流(stream)
    midi_stream = stream.Stream(output_notes)  # 把上面的循环输出结果传到流

    # 写入midi文件
    midi_stream.write('midi', fp='music/output.mid')  # 最终输出的文件名是output.mid，格式是mid


def train():
    notes = get_notes()
    # 得到所有不重复的音调数目
    num_pitch = len(set(notes))
    network_input, network_output = prepare_sequences(notes, num_pitch)
    model = network_model(network_input, num_pitch)
    # 输入，音符的数量，训练后的参数文件(训练的时候不用写)
    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"

    # 用checkpoint(检查点)文件在每一个Epoch结束时保存模型的参数
    # 不怕训练过程中丢失模型参数，当对loss损失满意的时候可以随时停止训练
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,  # 保存参数文件的路径
        monitor='loss',  # 衡量的标准
        verbose=0,  # 不用冗余模式
        save_best_only=True,  # 最近出现的用monitor衡量的最好的参数不会被覆盖
        mode='min'  # 关注的是loss的最小值
    )

    callbacks_list = [checkpoint]
    # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # 用fit方法来训练模型
    model.fit(network_input, network_output, epochs=10, batch_size=64, callbacks=callbacks_list)
    # 输入，标签（衡量预测结果的），轮数，一次迭代的样本数，回调
    # model.save(filepath='./model',save_format='h5')


def prepare_sequences(notes, num_pitch):
    # 从midi中读取的notes和所有音符的数量
    """
    为神经网络提供好要训练的序列
    """
    sequence_length = 100  # 序列长度

    # 得到所有不同音高的名字
    pitch_names = sorted(set(item for item in notes))
    # 把notes中的所有音符做集合操作，去掉重复的音，然后按照字母顺序排列

    # 创建一个字典，用于映射 音高 和 整数
    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_names))
    # 枚举到pitch_name中

    # 创建神经网络的输入序列和输出序列
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):  # 循环次数，步长为1
        sequence_in = notes[i:i + sequence_length]
        # 每次输入100个序列，每隔长度1取下一组，例如：(0,100),(1,101),(50,150)
        sequence_out = notes[i + sequence_length]
        # 真实值，从100开始往后
        network_input.append([pitch_to_int[char] for char in sequence_in])  # 列表生成式
        # 把sequence_in中的每个字符转为整数（pitch_to_int[char]）放到network_input
        network_output.append(pitch_to_int[sequence_out])
        # 把sequence_out的一个字符转为整数

    n_patterns = len(network_input)  # 输入序列长度

    # 将输入序列的形状转成神经网络模型可以接受的
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # 输入，要改成的形状

    # 将输入标准化，归一化
    network_input = network_input / float(num_pitch)
    # 将期望输出转换成{0，1}布尔矩阵，配合categorical_crossentrogy误差算法的使用
    network_output = tf.keras.utils.to_categorical(network_output)
    # keras中的这个方法可以将一个向量传进去转成布尔矩阵，供交叉熵的计算
    return network_input, network_output


def generate():
    # 加载用于训练神经网络的音乐数据
    # with open('F:/Codes/computer-code/code1/data/notes', 'rb') as filepath:  # 以读的方式打开文件
    # notes = pickle.loads(filepath)
    notes = get_notes()
    # 得到所有不重复的音符的名字和数目
    pitch_names = sorted(set(item for item in notes))
    num_pitch = len(set(notes))
    network_input, normalized_input = prepare_sequences1(notes, pitch_names, num_pitch)
    files = os.listdir()
    minloss = {}
    for i in files:
        if 'weights' in i:
            print(i)
            num = i[11:15]
            minloss[num] = i

    print(1)
    print(minloss)
    print(2)
    print(minloss.keys())
    print(3)
    print(min(minloss.keys()))
    print(4)
    print(minloss[min(minloss.keys())])
    best_weights = minloss[min(minloss.keys())]
    print('最佳模型文件为:' + best_weights)
    # 载入之前训练是最好的参数（最小loss），来生成神经网络模型
    model = network_model(normalized_input, num_pitch, best_weights)

    # 用神经网络来生成音乐数据
    prediction = generate_notes(model, network_input, pitch_names, num_pitch)

    # 用预测的音乐数据生成midi文件
    create_music(prediction)


def prepare_sequences1(notes, pitch_names, num_pitch):
    # 从midi中读取的notes和所有音符的数量
    """
    为神经网络提供好要训练的序列
    """
    sequence_length = 100  # 序列长度

    # 创建一个字典，用于映射 音高 和 整数
    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_names))
    # 枚举到pitch_name中

    # 创建神经网络的输入序列和输出序列
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):  # 循环次数，步长为1
        sequence_in = notes[i:i + sequence_length]
        # 每次输入100个序列，每隔长度1取下一组，例如：(0,100),(1,101),(50,150)
        sequence_out = notes[i + sequence_length]
        # 真实值，从100开始往后
        network_input.append([pitch_to_int[char] for char in sequence_in])  # 列表生成式
        # 把sequence_in中的每个字符转为整数（pitch_to_int[char]）放到network_input
        network_output.append([pitch_to_int[sequence_out]])
        # 把sequence_out的一个字符转为整数

    n_patterns = len(network_input)  # 输入序列长度

    # 将输入序列的形状转成神经网络模型可以接受的
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # 输入，要改成的形状

    # 将输入标准化，归一化
    normalized_input = normalized_input / float(num_pitch)
    return (network_input, normalized_input)


def generate_notes(model, network_input, pitch_names, num_pitch):
    """
    基于序列音符，用神经网络来生成新的音符
    """
    # 从输入里随机选择一个序列，作为“预测”/生成的音乐的起始点
    start = np.random.randint(0, len(network_input) - 1)  # 从0到神经网络输入-1中随机选择一个整数

    # 创建一个字典用于映射 整数 和 音调，和训练相反的操作
    int_to_pitch = dict((num, pitch) for num, pitch in enumerate(pitch_names))

    pattern = network_input[start]  # 随机选择的序列起点

    # 神经网络实际生成的音符
    prediction_output = []

    # 生成700个音符
    for note_index in range(700):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        # 输入，归一化
        prediction_input = prediction_input / float(num_pitch)

        # 读取参数文件，载入训练所得最佳参数文件的神经网络来预测新的音符
        prediction = model.predict(prediction_input, verbose=0)  # 根据输入预测结果

        # argmax取最大的那个维度（类似One-hot编码）
        index = np.argmax(prediction)
        result = int_to_pitch[index]
        prediction_output.append(result)

        # start往后移动
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return prediction_output


if __name__ == '__main__':
    # train()
    generate()
    transfer()
