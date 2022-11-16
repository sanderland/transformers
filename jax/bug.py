

from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

dataset = load_dataset("ethos", "binary")
BASE_MODELNAME = "sentence-transformers/all-MiniLM-L6-v2"

class Object:
    def __init__(self):
        config = AutoConfig.from_pretrained(BASE_MODELNAME)
        self.bert = AutoModel.from_config(config=config, add_pooling_layer=False)
        self.tok = AutoTokenizer.from_pretrained(BASE_MODELNAME)

    @staticmethod
    def tokenize(tok, examples):
        tokenized_texts = tok(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        return tokenized_texts

instance = Object()
def format(ex):
    return instance.tokenize(instance.tok, ex)

result = dict()
for phase in ["train"]:
    result[phase] = dataset[phase].map(format, batched=True, load_from_cache_file=True, num_proc=2)


