import torch
from torch import nn
from transformers import LlamaForCausalLM
from config import Config


class LlamaLoraModel(nn.Module):
    def __init__(self):
        super(LlamaLoraModel, self).__init__()
        model = LlamaForCausalLM.from_pretrained(
            Config.base_model,
            load_in_8bit=Config.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto"  # 设置为auto时会默认使用所有可以使用的gpu，并且将模型分片加载。
        )  # 权重类型是float16

        self.model.config.use_cache = False
        for name, param in model.named_parameters():
            if not ('29' in name or '30' in name or '31' in name or 'head' in name or 'embed' in name):
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels):
        output = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = output.loss
        return loss

