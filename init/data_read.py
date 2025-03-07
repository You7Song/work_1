import json  
from torch.utils.data import Dataset  
from transformers import AutoTokenizer
from collections import defaultdict

class DuReaderQG(Dataset):  
    def __init__(self, json_file, tokenizer_name="uer/t5-small-chinese-cluecorpussmall"):  
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # 读取 JSON 文件  
        with open(json_file, 'r', encoding='utf-8') as f: 
            # self.data =  [json.loads(line) for line in f]
            raw_data = [json.loads(line) for line in f]
        self.data = []
        # i = 0
        for qca in raw_data:
            # 问题答案token化
            qc = self.tokenizer(
                "question: "+qca['question']+" context: "+qca['context'],
                max_length=512,
                truncation=True,
            )
            a = self.tokenizer(
                qca['answer'],
                max_length=200,
                truncation=True,
            )

            self.data.append({
                'input_ids': qc['input_ids'],
                'attention_mask': qc['attention_mask'],
                'labels': a['input_ids'][1:] #不切片模型会学习到[CLS]后面输出的是[CLS]........
            })
        
    
    def __len__(self):  
        # 返回数据集的大小  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        return self.data[idx]



class DuReaderQGTest(Dataset):  
    def __init__(self, json_file, tokenizer_name="uer/t5-small-chinese-cluecorpussmall"):  
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  
        
        # 读取 JSON 文件  
        with open(json_file, 'r', encoding='utf-8') as f:  
            raw_data = [json.loads(line) for line in f]  
        
        # 使用字典来合并相同 context 和 question 的 answer  
        data_dict = defaultdict(lambda: {'input_ids': None, 'attention_mask': None, 'labels': []})  
        
        for qca in raw_data:  
            # 对 question 和 context 进行 token 化  
            qc = self.tokenizer(  
                "question: " + qca['question'] + " context: " + qca['context'],  
                max_length=512,  
                truncation=True,  
            )  
            
            # 对 answer 进行 token 化  
            a = self.tokenizer(  
                qca['answer'],  
                max_length=200,  
                truncation=True,  
            )  
            
            # 使用 context 和 question 的组合作为字典的键  
            key = (qca['context'], qca['question'])  
            
            # 如果该 context 和 question 组合尚未被处理，则初始化 input_ids 和 attention_mask  
            if data_dict[key]['input_ids'] is None:  
                data_dict[key]['input_ids'] = qc['input_ids']
            
            # 将 answer 的 token 化结果添加到 labels 中  
            data_dict[key]['labels'].append(a['input_ids'][1:])  # 去掉 [CLS] 标记  
        
        # 将字典转换为列表  
        self.data = list(data_dict.values())  
    
    def __len__(self):  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        return self.data[idx]
    
# # 使用示例  
# dataset = DuReaderQG('DuReaderQG/train.json')  

# i = 0
# for data in dataset:
#     print(dataset.tokenizer.convert_ids_to_tokens(data['labels']))
#     i += 1
#     if (i > 2):
#         break