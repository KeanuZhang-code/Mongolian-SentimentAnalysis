# 导入模块部分（按功能分组）
import json, re, os, time, random, jieba, pickle, shutil, datetime

import matplotlib
from jedi.api.refactoring import inline
from tqdm import tqdm
from collections import defaultdict, Counter

import numpy as np

np.random.seed(7)
import pandas as pd
import matplotlib.pyplot as plt

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys

if sys.platform.startswith('win'):
    plt.rcParams['font.family'] = ['sans-serif']  # windows系统
else:
    plt.rcParams["font.family"] = 'Arial Unicode MS'  # 苹果系统

from keras.engine.training import Model

# 文本预处理模块
from keras.preprocessing.text import Tokenizer

# 评估函数模块
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from keras.utils import to_categorical

# 网络层模块
from keras.layers import Dense, Dropout, Embedding, Input, Concatenate
from keras.layers import Conv1D
from keras.layers import GlobalMaxPool1D
from keras.preprocessing.sequence import pad_sequences  # 序列填充

# 优化器模块
from keras.optimizers import Adam

# 回调函数模块
from keras.callbacks import CSVLogger

# 工具模块
from keras.utils import to_categorical, plot_model, print_summary

import warnings

warnings.filterwarnings("ignore")

# GPU配置
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# tf2 GPU检测
import tensorflow as tf

gpus = tf.config.list_physical_devices(device_type='GPU')
print("Num GPUs Available: ", len(gpus))

# 全局参数
epochs = 20  # 批次
embedding_size = 128  # 嵌入维度
drop = 0.5  # 丢弃率
sample_size = 1000  # 一共取多少数据集
num_classes = 7  # 分类个数

# 模型目录初始化
import shutil

shutil.rmtree("model", ignore_errors=True)  # 删除test文件夹下所有的文件、文件夹
if not os.path.exists("model"):
    os.mkdir("model")


# -------------------- 文本处理函数 --------------------
def clean_str(string):
    """
    对数据集中的字符串做清洗.
    """
    string = str(string)
    # 去除标点符号
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", string)
    # 去除网页
    string = re.sub(r"<[^>].*?>|&.*?;", "", string)
    # 去除数字和英文
    string = re.sub("[a-zA-Z0-9]", "", string)
    # 去除非中文字符
    string = re.sub("[^\u4e00-\u9fff]", "", string)

    return string.strip().lower()


def get_texts_vector(texts=None, maxlen=None):
    """
    texts: 分好词的文本
    maxlen: 序列的最大长度 默认句子的最大长度 可以自己指定，
    返回texts的 索引填充后的向量 和 分词器 tokenizer（可以获得词汇表）

    # 使用
    # 填充序列  分词器  句子长度（预测需要用到）
    # sequences_pad,tokenizer,maxlen = get_texts_vector(texts)
    # word2num,num2word = tokenizer.word_index,tokenizer.index_word
    # 词汇表长度
    # word_num = len(num2word)
    """
    tokenizer = Tokenizer(
        num_words=None,  # 低于该数值 删除
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',  # 过滤字符
        lower=True,  # 是否小写
        split='',  # 分隔符
        char_level=True,  # true 切分字 false 切分词
        oov_token='<UNK/>',  # 未登陆词设置
    )

    # 训练分词器
    # texts 可以是[[word,word],,,] 也可以是["word word", "word word" ] 但是分好词
    tokenizer.fit_on_texts(texts)
    word2num = tokenizer.word_index
    num2word = tokenizer.index_word

    # 文本的索引向量化 ,可以结合pad_sequences向量化
    texts_vector = tokenizer.texts_to_sequences(texts)

    # 如果不传入最大长度，默认句子的长度的最大值
    if not maxlen:
        maxlen = max([len(_) for _ in texts_vector])

    # 序列填充
    sequences_pad = pad_sequences(
        texts_vector,  # 二维列表 [[ num1,num2],,]
        maxlen=maxlen,  # 序列长度
        padding='post',  # 长度低于maxlen时  pre 序列前补齐  post序列后补齐
        truncating='post',  # 长度超出maxlen时 pre 序列前截断 post 序列后截断
        value=0.0,  # 补齐的值
        dtype='int32',
    )
    return sequences_pad, tokenizer, maxlen


# -------------------- 模型构建函数 --------------------
def build_cnn_model():
    # 输入层
    inputs = Input(shape=(maxlen,))

    # 嵌入层
    embedding_layer = Embedding(
        word_num,  # 单词维度
        embedding_size,  # embedding_size
        input_length=maxlen  # 文本向量长度
    )(inputs)

    # 定义多个卷积核的大小
    filter_sizes = [3, 4, 5]
    num_filters = 64

    # 存储所有池化后的特征
    pooled_outputs = []

    # 对每个卷积核应用Conv1D和GlobalMaxPooling1D，并存储结果
    for filter_size in filter_sizes:
        conv_layer = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(embedding_layer)
        pooled_layer = GlobalMaxPool1D()(conv_layer)
        pooled_layer = Dropout(drop)(pooled_layer)
        pooled_outputs.append(pooled_layer)

    # 拼接所有池化后的特征
    concatenated = Concatenate()(pooled_outputs)

    # 全连接层
    dense_layer = Dense(units=256, activation='relu')(concatenated)
    dense_layer = Dropout(drop)(dense_layer)

    # 输出层
    output_layer = Dense(units=num_classes, activation='softmax')(dense_layer)

    # 构建模型
    model = Model(inputs=inputs, outputs=output_layer)
    return model


# -------------------- 评估与可视化函数 --------------------
# 评估函数
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report,confusion_matrix


def evaluate_fun(true_y, pred_y):
    """
    评估函数指标
    """
    acc = accuracy_score(true_y, pred_y)
    f1 = f1_score(true_y, pred_y, average="macro")
    pre = precision_score(true_y, pred_y, average="macro")
    recall = recall_score(true_y, pred_y, average="macro")
    return {
        'acc': acc,
        'f1': f1,
        'pre': pre,
        'recall': recall,
    }


def plot_train(history, model_name):
    """
    绘制模型训练过程
    """
    print(history.history.keys())

    # 绘制训练Loss曲线
    plt.figure(1, figsize=(12, 6), dpi=100)
    plt.plot(history.history['loss'], label='训练集')
    plt.plot(history.history['val_loss'], label='测试集')
    plt.title(f'{model_name} loss 曲线')
    plt.legend()
    plt.show()

    # 绘制训练acc曲线
    plt.figure(1, figsize=(12, 6), dpi=100)
    plt.plot(history.history['acc'], label='训练集')
    plt.plot(history.history['val_acc'], label='测试集')
    plt.title(f'{model_name} 准确率 曲线')
    plt.legend()
    plt.show()


def plot_confusion_matrix(true_y, pred_y, labels=[0, 1, 2], confusion_matrix=None):
    """
    绘制混淆矩阵
    依次是真实值  预测值  标签列表（可以为中文 也可以为数字）
    """
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(true_y, pred_y, sample_weight=None)

    plt.rcParams['figure.dpi'] = 200  # 分辨率
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)  # 可以改变颜色
    indices = range(conf_matrix.shape[0])

    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('真实值')
    plt.ylabel('预测值')

    # 显示数据
    for first_index in range(conf_matrix.shape[0]):  # trues
        for second_index in range(conf_matrix.shape[1]):  # preds
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.title(f" 混淆矩阵")
    plt.show()


# -------------------- 数据加载与处理 --------------------
# 从数据集加载数据　　不会改变数据集类型
# 最终的特征为 label text
def get_df_from_txt(data_dir):
    """
    从csv获取文本数据
    并且类别转化为数字
    data = get_df_csv('../data/weibo_senti_100k.csv',balance=1000)
    """
    data_dir = "data/Mongolian_Datasets"

    final_df = pd.DataFrame()
    for filename in os.listdir(data_dir):
        label = filename[:-4]  # 情感类别
        # 不是数据集 不要
        file_path = data_dir + '/' + filename
        if not file_path.endswith('.txt'):
            continue

        texts = []
        labels = []
        tmp_df = pd.DataFrame()

        # 遍历该类别数据集
        with open(file_path, 'r', encoding="utf8") as f:
            for line in f.readlines():
                text = line.strip().split(";")[1:]
                text = ";".join(text)
                texts.append(text)
                labels.append(label)
            tmp_df['text'] = texts
            tmp_df['label'] = labels
        print(f"{label} 数据集长度为:{len(tmp_df)}")

        # 合并数据集
        final_df = pd.concat([final_df, tmp_df])

    final_df = final_df.rename(columns={"review": 'text'})
    final_df.index = range(len(final_df))

    return final_df

# 加载数据
data_dir = "data/Mongolian_Datasets"
data = get_df_from_txt(data_dir)
data.head()

# 标签统一规定都是字符串
print(data['label'].dtype)
# 这一步知识把数字转化为 中文
data.head()

data.shape

# -------------------- 文本向量化处理 --------------------
# 保存训练的所有信息
allInfo_path = 'model/train_info.pkl'

if os.path.exists(allInfo_path):
    X_train, Y_train, index2label, tokenizer, maxlen = pickle.load(open(allInfo_path, 'rb'))
    word2num, num2word = tokenizer.word_index, tokenizer.index_word
    word_num = len(num2word) + 1  # 必须加1
else:
    # 数字对应的标签
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    new_labels = le.fit_transform(data["label"])
    data["label"] = new_labels

    index2label = {i: j for i, j in enumerate(le.classes_)}

    # 分词后的文本
    texts = data["text"]
    # 填充序列  分词器  句子长度（预测需要用到）
    sequences_pad, tokenizer, maxlen = get_texts_vector(texts)
    word2num, num2word = tokenizer.word_index, tokenizer.index_word

    # 词汇表长度
    word_num = len(num2word) + 1  # 必须加1
    X_train, Y_train = sequences_pad, data['label']
    pickle.dump([X_train, Y_train, index2label, tokenizer, maxlen], open(allInfo_path, 'wb'))

print("文本处理词向量成功！")

# 保存 类别信息  tokenizer 句子的最大长度 词汇表  预测时需要
pickle.dump([index2label, tokenizer, maxlen, word2num], open('model/final_info.pkl', 'wb'))

X_train.shape

# -------------------- 数据集划分 --------------------
from keras.utils import to_categorical

# 转化为类别 为独热编码
y_binary = to_categorical(Y_train, num_classes=num_classes)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train, y_binary, test_size=0.2, shuffle=True)
print("x_train 的形状为：", x_train.shape)
print("y_train 的形状为：", y_train.shape)
print("x_test 的形状为：", x_test.shape)
print("y_test 的形状为：", y_test.shape)

# -------------------- 模型训练 --------------------
# 构建模型
model = build_cnn_model()

# 训练
begin_time = time.time()

# 训练轮结果保存到csv文件
csvlogger = CSVLogger(
    filename='logs/training_log.csv',  # 保存文件
    separator=',',  # 分隔符 默认为 ，
    append=False  # true 追加  false 覆盖
)

callbacks_list = [csvlogger, ]

# 优化器
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',  # binary_crossentropy　categorical_crossentropy
              metrics=['acc']
              )

# 开始训练
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=epochs,
                    #                 validation_split=0.2, # 如果没有validation_data 自动从训练集划分测试集
                    validation_data=[x_test, y_test],
                    callbacks=callbacks_list
                    )

# 保存模型
model.save('model/model.h5')  # 创建 HDF5 文件 'my_model.h5'
model.save_weights('model/model_weights.h5')

# -------------------- 结果评估与可视化 --------------------
# 结果可视化
plot_train(history, 'cnn')

# 模型评估
# 获取测试集的真实标签（将独热编码转换为原始标签）
true_y = np.argmax(y_test, axis=1)
# 获取模型对测试集的预测标签
pred_y = model.predict(x_test).argmax(axis=1)

# 输出评价指标
print("f1 recall precision acc 指标：")
print(evaluate_fun(true_y, pred_y))

print("分类报告：")
true_y_ch = [index2label[i] for i in true_y]
pred_y_ch = [index2label[i] for i in pred_y]
print(classification_report(true_y_ch, pred_y_ch))

# -------------------- 文本预测函数 --------------------
import json, re, os, time, random, jieba, pickle, shutil, datetime
from keras.preprocessing.sequence import pad_sequences  # 序列填充
from keras.models import Sequential, Model, save_model, load_model


def predict_text_label(text):
    """
    预测一个文本的类别
    """
    # 加载信息
    num2label, tokenizer, maxlen, word2num = pickle.load(open('model/final_info.pkl', 'rb'))
    # 加载模型
    model = load_model('model/model.h5')

    # 分词
    texts = [text, ]
    text_vector = tokenizer.texts_to_sequences(texts)
    sequences_pad = pad_sequences(
        text_vector,  # 二维列表 [[ num1,num2],,]
        maxlen=maxlen,  # 序列长度
        padding='post',  # 长度低于maxlen时  pre 序列前补齐  post序列后补齐
        truncating='post',  # 长度超出maxlen时 pre 序列前截断 post 序列后截断
        value=0.0,  # 补齐的值
        dtype='int32',
    )
    #     label = model.predict_classes(sequences_pad)[0][0]
    label = model.predict(sequences_pad).argmax(axis=1)[0]
    print(label)
    #     print(label)
    labelch = num2label[label]
    print(f"预测的文本：{text}")
    print("---" * 45)
    print(f"数字标签：{label} , 中文标签：{labelch}")
    print("\n")


# 测试预测函数
text = "ᠬᠤᠳᠠᠯᠳᠤᠭᠰᠠᠨ ᠤ᠋ ᠳᠠᠷᠠᠭᠠᠬᠢ ᠦᠢᠯᠡᠴᠢᠯᠡᠭᠡ ᠬᠤᠳᠠᠯᠳᠤᠨ ᠠᠪᠬᠤ ᠠ᠋ᠴᠠ ᠡᠮᠦᠨ᠎ᠡ ᠬᠤᠳᠠᠯᠳᠤᠨ ᠠᠪᠬᠤ ᠣᠪᠣᠷ ᠨᠢ ᠪᠦᠷ ᠠᠳᠠᠯᠢ ᠦᠭᠡᠢ ᠪᠣᠯᠵᠠᠢ ᠂ ᠰᠡᠭᠦᠯᠡᠷ ᠪᠦᠷ ᠬᠤᠳᠠᠯᠳᠤᠨ ᠠᠪᠬᠤ ᠦᠭᠡᠢ ᠪᠣᠯᠤᠨ᠎ᠠ  "
predict_text_label(text)