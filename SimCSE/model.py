import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import transformers

transformers.logging.set_verbosity_error()

class TextBackbone(torch.nn.Module):
    def __init__(self, path='/hy-tmp/bert', output_dim=128):
        super(TextBackbone, self).__init__()
        self.bert = BertModel.from_pretrained(path).cuda()
        self.drop = torch.nn.Dropout(p=0.1)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.drop(output.pooler_output)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def predict(self, x):
        x['input_ids'] = x['input_ids'].squeeze(1)
        x['attention_mask'] = x['attention_mask'].squeeze(1)
        x['token_type_ids'] = x['token_type_ids'].squeeze(1)
        out = self.bert(**x)
        out = self.fc(out.pooler_output)
        out = F.normalize(out, p=2, dim=-1)
        return out