# 导入若干包
from __future__ import absolute_import, division, print_function
import logging
import numpy as np
import os
import random
import sys
import time
import torch

# 用于设定全局配置的命名空间
from argparse import Namespace

# 导入常用的模型处理工具
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

# 导入进度条可视化工具
from tqdm import tqdm

# 从transformers中导入BERT模型的相关工具
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer)

# 从transformers中导入GLUE数据集评估指标方法
from transformers import glue_compute_metrics as compute_metrics

# 从transformers中导入GLUE数据集的输出模式
from transformers import glue_output_modes as output_modes

# 从transformers中导图GLUE数据集的预处理器processors
from transformers import glue_processors as processors

# 从transformers中导入GLUE数据集中的特征处理器
from transformers import glue_convert_examples_to_features as convert_examples_to_features

# 设定与日志打印相关的配置
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

# print("torch version:", torch.__version__)

# 设置torch容许启动的线程数量
torch.set_num_threads(1)
# print(torch.__config__.parallel_info())

# 实例化一个配置的命名空间
configs = Namespace()

# 设定模型的输出文件路径
configs.output_dir = "./bert_finetuning_test/"

# 设定验证数据集所在的路径
configs.data_dir = "./glue_data/MRPC"

# 设定预训练模型的名称
configs.model_name_or_path = "bert-base-uncased"

# 设定文本的最大长度
configs.max_seq_length = 128

# GLUE中的任务名称(需要小写)
configs.task_name = "MRPC".lower()

# 取出预处理器
configs.processor = processors[configs.task_name]()

# 得到对应模型的输出模式
configs.output_mode = output_modes[configs.task_name]

# 得到该任务的对应标签种类列表
configs.label_list = configs.processor.get_labels()

# 定义模型的类型
configs.model_type = "bert".lower()

# 设定是否全部使用小写文本
configs.do_lower_case = True

# 设定使用的设备
configs.device = "cpu"

# 设定每次验证的批次大小
configs.eval_batch_size = 8

# 设定GPU的数量
configs.n_gpu = 0

# 设定是否需要重写数据缓存
configs.overwrite_cache = False

# 编写一个随机种子的设置函数
def set_seed(seed):
    # seed: 代表的是种子整数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# 加载BERT预训练模型的数值映射器
tokenizer = BertTokenizer.from_pretrained(configs.output_dir, do_lower_case=configs.do_lower_case)

# 加载带有文本分类头的模型
model = BertForSequenceClassification.from_pretrained(configs.output_dir)

# 将模型传到设备上
model.to(configs.device)

# 构建评估函数
def evaluate(args, model, tokenizer):
    # args: 代表模型全局配置参数
    # model: 代表预训练的模型
    # tokenizer: 代表文本数据的数值映射器
    # 将任务名称和模型地址设定好
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    try:
        # 调用辅助函数, 加载原始或者已经缓存的数据, 得到一个验证数据集的迭代器对象
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer)

        # 判断模型的输出路径文件夹是否存在, 不存在的话提前创建
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # 实例化一个顺序采样器, 保证数据是顺序采集, 不改变原有数据的顺序, 一次提取数据
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # 设置若干日志信息
        logger.info("****** Running evaluation ******")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        # 初始化验证损失
        eval_loss = 0.0
        # 初始化验证步数
        nb_eval_steps = 0
        # 初始化预测的概率分布
        preds = None
        # 初始化输出的真实标签值
        out_label_ids = None

        # 循环遍历验证集数据, 采用进度条显示进度
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 开启评估模式
            model.eval()
            # 从batch数据中取出所有的相关信息存入元组
            batch = tuple(t.to(args.device) for t in batch)
            # 进行模型的评估
            with torch.no_grad():
                # 将batch中的数据分别封装进字典类型中
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}

                # 将字典作为参数传入模型中, 进行预测
                outputs = model(**inputs)
                # 输出的信息包含损失值和预测分布
                tmp_eval_loss, logits = outputs
                # 将损失值进行累加
                eval_loss += tmp_eval_loss.mean().item()
            # 验证步数进行累加
            nb_eval_steps += 1

            # 如果是批次数据的第一次循环赋值
            if preds is None:
                preds = logits.numpy()
                out_label_ids = inputs['labels'].numpy()
            else:
                # 将第二次和之后的预测结果, 以及真实标签添加到结果列表中
                preds = np.append(preds, logits.numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].numpy(), axis=0)

        # 跳出循环后, 当前轮次评估完毕
        eval_loss = eval_loss / nb_eval_steps
        # 取结果分布中最大的值对应的索引
        preds = np.argmax(preds, axis=1)
        # 评估指标结果
        result = compute_metrics(eval_task, preds, out_label_ids)
        # 在日志中打印每一轮的结果
        logger.info("****** Eval results {} ******")
        logger.info(str(result))
    except Exception as e:
        print(e)
    return result

# 构建加载缓存数据的函数
def load_and_cache_examples(args, task, tokenizer):
    # args: 代表全局配置参数
    # task: 代表任务名称
    # tokenizer: 代表数值映射器
    # 根据任务名称获得对应的数据预处理器
    processor = processors[task]()
    # 获取输出模式
    output_mode = output_modes[task]
    # 定义缓存数据文件的名称
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format('dev',
                                                        list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                                        str(args.max_seq_length),
                                                        str(task)))

    # 判断缓存文件是都存在
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # 直接使用torch.load加载缓存文件数据
        features = torch.load(cached_features_file)
    else:
        # 如果没有缓存数据文件, 需要使用预处理器从原始数据文件中加载数据
        examples = processor.get_dec_examples(args.data_dir)
        # 获取对应的标签
        label_list = processor.get_labels()
        # 生成模型需要的输入形式
        features = convert_examples_to_features(examples, tokenizer, label_list=label_list, max_length=args.max_seq_length,
                                                output_model=output_mode, pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
        # 添加日志信息
        logger.info("Saving features into cached file %s", cached_features_file)
        # 将特征保存成缓存文件
        torch.save(features, cached_features_file)

    # 将数据封装进TensorDataset中
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


# 使用torch.quantization.quantize_dynamic获取动态量化的模型
# 明确量化的对象是网络层为素有的nn.Linear的权重, 使其类型成为qint8
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# print(quantized_model)

# 编写打印模型占用磁盘空间尺寸的函数
def print_size_of_model(model):
    # model: 待评估大小的模型
    # 第一步: 将模型保存在磁盘上
    torch.save(model.state_dict(), "temp.p")
    # 第二步: 打印持久化文件的大小, 按照MB的单位进行打印
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    # 第三步: 移除临时文件
    os.remove('temp.p')

# 依次调用打印原始模型和动态量化模型的大小
# print_size_of_model(model)
# print_size_of_model(quantized_model)

# 编写评估耗时的打印模型表现的函数
def time_model_evaluation(model, configs, tokenizer):
    # model: 代表待评估的模型
    # configs: 设置好的参数空间
    # tokenizer: 设置好的数值映射器
    # 第一步获取开始时间
    eval_start_time = time.time()
    # 第二步进行模型评估
    result = evaluate(configs, model, tokenizer)
    # 第三步获取结束时间
    eval_end_time = time.time()
    # 计算出评估的耗时
    eval_duration_time = eval_start_time - eval_end_time
    # 打印模型评估的结果
    print("Evaluation result:", result)
    # 打印评估的耗时
    print("Evaluation total time (seconds): {0:.1f}".format(eval_duration_time))


# 依次调用评估函数得到原始模型和动态量化模型的耗时
# time_model_evaluation(model, configs, tokenizer)
# time_model_evaluation(quantized_model, configs, tokenizer)

# 设定量化模型的保存路径
quantized_output_dir = configs.output_dir + "quantized/"
# 首先判断路径是否存在, 如果不存在创建出来, 并保存模型
if not os.path.exists(quantized_output_dir):
    os.makedirs(quantized_output_dir)
    quantized_model.save_pretrained(quantized_output_dir)

