import os
import jieba
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.python.keras.utils import to_categorical



# 读取数据
def get_data():
    path = '..//分类数据集'
    files = os.listdir(path)
    data = []
    for filename in files:
        text = os.listdir(path+'//'+filename)
        # 打开文件夹
        text_data = []
        for text_files in text:
            # 打开txt文件
            try:
                with open(path+'//'+filename+'//'+text_files, encoding='GB18030') as text_context:
                    line = text_context.readlines()
                    string = ''

                    for i in line:
                        a = i.strip("\n")
                        b = a.strip("\u3000")
                        c = b.strip("&nbsp;")
                        string += c

                    text_data.append(string)
            except Exception as e:
                print(filename+text_files+'出现编码错误')
        data.append(text_data)

    return files, data


# 数据预处理
def data_pre(data):
    # 得到标签
    label = [[i]*len(data[i]) for i in range(len(data))][0]
    label = to_categorical(label)
    # 切词
    context = []
    for i in data:
        for j in i:
            context.append(jieba.lcut(j))

    # 构建词典
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(context)

    train_tags_title = tokenizer.texts_to_sequences(context)
    train_tags_title_preprocessed = pad_sequences(train_tags_title, maxlen=45, padding='post')

    # 预训练词向量
    # embedding_matrix = np.zeros((278028, 30), dtype=np.float32)
    # f = open('wiki.zh.text.vector', encoding='utf-8')
    # f = f.readlines()
    # for text in f:
    #     text = text.split()
    #     if text[0] in context:
    #         embedding_matrix[context[text[0]]] = text[1:]

    # 模型
    x_1 = Input(shape=(45,))  # 输入数据维度
    embed_1 = Embedding(input_dim=45, output_dim=45)(x_1)  # 将索引值转化为稠密向量，且只能做第一层
    L_1 = (LSTM(64))(embed_1)  # 第一个括号构建一个层 64是输出空间的维度，第二个括号用该层做计算
    L_1 = Dropout(0.5)(L_1)  # 防止过拟合，0.5在这里是需要丢弃的输入比例
    L_1 = Dense(9, activation='softmax')(L_1)  # 3是输出空间维度
    model_one = Model(x_1, L_1)  # x_1输入，L_1输出
    model_one.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])  # 'binary_crossentropy'
    history = model_one.fit(train_tags_title_preprocessed, label, batch_size=512, epochs=20, validation_split=0.1, shuffle=True)
    # 汇总acc函数历史数据
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # 汇总损失函数历史数据
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    files, data = get_data()
    data_pre(data)