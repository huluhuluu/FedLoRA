from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
from torch import nn

llama_cache = '/data/huluhulu/fedbertlora/cache/llama-2-7b-hf'
cache_path = '/data/huluhulu/fedbertlora/cache'
model_id = 'meta-llama/Llama-2-7b-hf'
task = 'cola'
local_bs, lr = 1, 0.01
lora_rank, lora_alpha = 8, 16
task_to_keys = {
    "banking77": ("text", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
gpu = 0
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(llama_cache)
# new token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# set pad token
tokenizer.pad_token = '[PAD]'

class DatasetSplit(Dataset):
    def __init__(self, dataset, task, idxs):
        self.dataset = dataset
        self.task = task
        self.idxs = [int(idx) for idx in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        global tokenizer, task_to_keys
        if task_to_keys[self.task][1] is not None:
            text_encoded = tokenizer(self.dataset[self.idxs[item]][task_to_keys[self.task][0]], self.dataset[self.idxs[item]][task_to_keys[self.task][1]],  truncation=True, padding='max_length', return_tensors='pt')
        else:
            text_encoded = tokenizer(self.dataset[self.idxs[item]][task_to_keys[self.task][0]], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        
        input_ids = text_encoded['input_ids'].squeeze()
        attention_mask = text_encoded['attention_mask'].squeeze()
        # print(input_ids, attention_mask, self.dataset[self.idxs[item]]['label'])
        return input_ids, attention_mask, self.dataset[self.idxs[item]]['label']

# data
data = load_dataset("nyu-mll/glue", task, cache_dir = cache_path + '/data')
print(data)
data_train = data['train']
num_labels = len(data_train.features["label"].names)
dataloader = DataLoader(DatasetSplit(data_train, task , [i for i in range(len(data_train))]), batch_size=local_bs, shuffle=True)

# model = AutoModelForCausalLM.from_pretrained(cache_path, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(llama_cache, num_labels=num_labels)

print(model)
print('------------------------------')
for name, param in model.named_parameters():
    print(name)
exit()
 # Define LoRA Config
lora_config = LoraConfig(
    r = lora_rank,
    lora_alpha = lora_alpha,
    target_modules= ["q_proj", "v_proj"],
    lora_dropout = 0.05,
    bias = "none"
)
# apply lora to bert model
model = get_peft_model(model, lora_config)
print(model)
print('------------------------------')
model.print_trainable_parameters()
print('------------------------------')
# train once
input_ids, attention_mask, label = next(iter(dataloader))
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True  # 包括调用栈信息，有助于分析性能瓶颈
) as prof:
    # prepare
    model.to(device)
    model.eval()
    input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
    model.zero_grad()
    # forward
    logits = model(input_ids, attention_mask=attention_mask).logits  # Get logits from model output
    loss = loss_func(logits, label)
    # backward
    # loss.backward()
    # optimizer.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace('./resnet_profile.json')