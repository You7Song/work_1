from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, T5ForConditionalGeneration, Seq2SeqTrainingArguments
from init.data_read import DuReaderQG
from transformers import Seq2SeqTrainer
import numpy as np
import logging 
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import TrainerCallback  
# 创建一个自定义过滤器，只过滤掉特定的日志消息  
class TokenizerDeprecationFilter(logging.Filter):  
    def filter(self, record):  
        # 当日志消息包含这段字符串时，返回 False（不输出日志）  
        return "Trainer.tokenizer is now deprecated" not in record.getMessage()  
# 获取 Trainer 模块对应的 logger  
logger = logging.getLogger("transformers.trainer")  
# 添加过滤规则  
logger.addFilter(TokenizerDeprecationFilter()) 



def compute_metrics(eval_pred):  
    predictions, labels = eval_pred  
    # 将-100替换为0，因为这是padding的标记  
    labels = np.where(labels != -100, labels, 0)  
    
     # 使用 batch_decode 一次性解码整个批次  
    predictions = dataset_train.tokenizer.batch_decode(predictions, skip_special_tokens=True)  
    labels = dataset_train.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 将标签转换为列表形式，因为 corpus_bleu 需要列表形式的参考文本  
    labels = [[label.split()] for label in labels]  
    predictions = [pred.split() for pred in predictions]
    
    # 使用平滑函数  
    smoothie = SmoothingFunction().method1
    
    # 计算 BLEU 分数  
    bleu1 = corpus_bleu(labels, predictions, weights=(1, 0, 0, 0), smoothing_function=smoothie)  
    bleu2 = corpus_bleu(labels, predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)  
    bleu3 = corpus_bleu(labels, predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)  
    bleu4 = corpus_bleu(labels, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return {  
        "eval_BLEU-1": bleu1,  
        "eval_BLEU-2": bleu2,  
        "eval_BLEU-3": bleu3,  
        "eval_BLEU-4": bleu4,  
    }
    


# 初始化数据集以及模型
dataset_train = DuReaderQG('DuReaderQG/train.json')  
dataset_eval = DuReaderQG('DuReaderQG/dev.json')  

# for i in range(5):  # 打印前5个样本  
#     print(dataset_train[i])
#     break

model = T5ForConditionalGeneration.from_pretrained("uer/t5-small-chinese-cluecorpussmall")

# 设置数据收集器
data_collator = DataCollatorForSeq2Seq(  
    tokenizer=dataset_train.tokenizer,   
    model=model,   
    padding=True
)

# 配置训练参数  
batch_size = 16
steps_per_epoch = len(dataset_train) // batch_size
training_args = Seq2SeqTrainingArguments(  
    output_dir="./t5_model_results",          # 输出目录  
    logging_dir='./logs',                     # 日志目录
    
    save_strategy="steps",                    # 保存策略  
    evaluation_strategy="steps",              # 评估策略  
    num_train_epochs=100,                       # 训练的轮数 
    logging_steps=50,
    
    per_device_train_batch_size=batch_size,            # 每个设备的训练 batch size  
    per_device_eval_batch_size=batch_size,             # 每个设备的评估 batch size  
    
 
  
    load_best_model_at_end=True,              # 在训练结束时加载最好的模型  
    metric_for_best_model="eval_BLEU-4",      # 用于评估的最佳模型指标  
    greater_is_better=True,                   # 指定更大的指标值是更好的模型  
    predict_with_generate=True,               # 在评估时使用生成模式  
    generation_num_beams=4,                   # 设置束搜索的数量  
    generation_max_length=50,                 # 最大生成长度
    warmup_steps=1500,                        # 设置warmup步数
    lr_scheduler_type="cosine_with_min_lr",   # 选择调度器
    learning_rate=5e-5,                       # 学习率  
    weight_decay=0.01,                        # 权重衰减
    lr_scheduler_kwargs={'num_cycles': 4, 'min_lr': 1e-6},
    
    
    save_steps=steps_per_epoch * 10,
    eval_steps=steps_per_epoch * 10,
) 

# 创建 Trainer 对象  
trainer = Seq2SeqTrainer(  
    model=model,  
    args=training_args,  
    train_dataset=dataset_train,  
    eval_dataset=dataset_eval,
    data_collator=data_collator,  # 使用 DataCollatorForSeq2Seq  
    compute_metrics=compute_metrics,
    processing_class=dataset_train.tokenizer,
    # callbacks=[DebugCallback()],
)



# 启动训练  
trainer.train()