from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer

question = "柚子茶的热量大概是多少?"
context = "（韩国）蜂蜜柚子茶的热量(以100克可食部分计)是305大卡(1276千焦)，在同类食物中单位热量较高。每100克（韩国）蜂蜜柚子茶的热量约占中国营养学会推荐的普通成年人保持健康每天所需摄入总热量的13%。"

# 加载分词器  
tokenizer = AutoTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")  

# print(tokenizer.special_tokens_map)

# 输入文本  
input_text = "question: " + question + " context: " + context

# 对输入文本进行编码  
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 加载模型  
model = T5ForConditionalGeneration.from_pretrained('t5_model_results/checkpoint-90700')  

# 生成输出  
output_ids = model.generate(  
    input_ids,  
    max_new_tokens=20,  # 设置生成的最大 token 数量  
    num_beams=4,        # 使用 beam search  
    early_stopping=True, # 提前停止  
    no_repeat_ngram_size=1
)

# 查找第一个 [SEP] 的 token ID  
sep_token_id = tokenizer.sep_token_id  

output = output_ids[0].tolist()
# 查找第一个 [SEP] 的位置  
try:  
    sep_index = output.index(sep_token_id)  
    truncated_ids = output[:sep_index]  # 截断到第一个 [SEP]  
except ValueError:  
    truncated_ids = output  # 如果没有 [SEP]，返回完整列表

# print(truncated_ids)
# 解码生成的文本  
output_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)


print(output_text)