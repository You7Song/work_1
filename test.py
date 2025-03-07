from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import T5ForConditionalGeneration
from init.data_read import DuReaderQGTest
import torch
from tqdm import tqdm
# 将模型移动到 GPU 0  
device = torch.device("cuda:0")  # 明确使用 gpu0  
# 加载模型  
model = T5ForConditionalGeneration.from_pretrained('t5_model_results/checkpoint-81630').to(device)
dataset_test = DuReaderQGTest('DuReaderQG/dev.json')
# 查找第一个 [SEP] 的 token ID  
sep_token_id = dataset_test.tokenizer.sep_token_id



predictions = []
labels = []
# 使用 tqdm 包装循环，显示进度条  
for data in tqdm(dataset_test, desc="Processing"):
    input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to(device)
    # 生成输出  
    output_ids = model.generate(  
        input_ids,  
        max_new_tokens=20,  # 设置生成的最大 token 数量  
        num_beams=4,        # 使用 beam search  
        early_stopping=True, # 提前停止  
        no_repeat_ngram_size=1
    )
    output = output_ids[0].tolist()
    try:  
        sep_index = output.index(sep_token_id)  
        output = output[:sep_index+1]  # 截断到第一个 [SEP]  
    except ValueError:  
        output = output  # 如果没有 [SEP]，返回完整列表
        
    output_text = dataset_test.tokenizer.decode(output, skip_special_tokens=True)
    predictions.append(output_text.split())
    
    cur_label = []
    for label in data['labels']:
        cur_label.append(dataset_test.tokenizer.decode(label, skip_special_tokens=True).split())
    labels.append(cur_label)
    

# 使用平滑函数  
smoothie = SmoothingFunction().method1

# 计算 BLEU 分数  
bleu1 = corpus_bleu(labels, predictions, weights=(1, 0, 0, 0), smoothing_function=smoothie)  
bleu2 = corpus_bleu(labels, predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)  
bleu3 = corpus_bleu(labels, predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)  
bleu4 = corpus_bleu(labels, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

results = {  
    "eval_BLEU-1": bleu1,  
    "eval_BLEU-2": bleu2,  
    "eval_BLEU-3": bleu3,  
    "eval_BLEU-4": bleu4,  
}

print(results)