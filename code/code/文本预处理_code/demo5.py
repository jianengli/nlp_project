# 导入相关的工具包
import torch
import torchtext
# 导入数据集中的文本分类任务
from torchtext.datasets import text_classification
import os

# 定义数据下载路径, 当前文件夹下的data文件夹
load_data_path = "./data"
if not os.path.isdir(load_data_path):
    os.mkdir(load_data_path)

# 选取torchtext包中的文本分类数据集'AG_NEWS', 即新闻主题分类数据
# 顺便将数据加载到内存中
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path)

