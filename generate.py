from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer

question = "白浅第几集要回眼睛"
context = "第56集 白浅恢复记忆便直接上了九重天找到素锦,素锦知道眼前的白浅就是当年的素素,惊吓不已,白浅冷笑望着跪地求饶的素锦道,本上神这眼睛你这三百年来用的可好,如今也该物归原主了吧,是你自己剜还是本上神亲自动手,素锦吓得连连后退,白浅取回来双眼,看都不看素锦一眼便离开了"


# 加载分词器  
tokenizer = AutoTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")  

# print(tokenizer.special_tokens_map)

# 输入文本  
input_text = "question: " + question + " context: " + context

# 对输入文本进行编码  
input_ids = tokenizer.encode(input_text, return_tensors='pt')
print(input_ids.shape)

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

print(truncated_ids)
# 解码生成的文本  
output_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)


print(output_text)