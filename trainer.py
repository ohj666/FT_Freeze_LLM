import os
import time
import torch
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup
from transformers import LlamaTokenizer
from model import LlamaLoraModel
from data_helper import DataHelper, LlamaDataset, collate_fn
from config import Config
from transformers import CONFIG_NAME, WEIGHTS_NAME
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Trainer:
    def __init__(self):
        # 训练次数
        self.epochs = Config.epochs
        # 热身 先进行低学习率再慢慢提升，这里设置预热步数
        self.warmup_steps = Config.warmup_steps

        self.learning_rate = Config.learning_rate
        # 权重衰减
        self.weight_decay = Config.weight_decay

        self.batch_size = Config.batch_size
        self.output = './models'
        self.tokenizer = LlamaTokenizer.from_pretrained(Config.base_model)

        self.train_data_loader, self.valid_data_loader = self.get_data_loader()
        print("get data loader done")

        # 初始化模型对象
        self.model = LlamaLoraModel()
        self.model = self.model.cuda()
        self.model.train()

        print("model load done")

        # for name, param in self.model.named_parameters():
        #     print(name, param.dtype)

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=self.optimizer,
                                                                   num_warmup_steps=self.warmup_steps,
                                                                   lr_end=0.0)

    def get_data_loader(self):
        # 加载数据集
        data_obj = DataHelper()
        train_data, valid_data = data_obj.gen_data()
        print("train data size: {}".format(len(train_data)))
        print("valid data size: {}".format(len(valid_data)))
        train_data_set = LlamaDataset(self.tokenizer, train_data)
        valid_data_set = LlamaDataset(self.tokenizer, valid_data)
        train_data_loader = DataLoader(train_data_set, batch_size=self.batch_size, drop_last=True,
                                       shuffle=True, collate_fn=collate_fn)
        valid_data_loader = DataLoader(valid_data_set, batch_size=self.batch_size, collate_fn=collate_fn)

        return train_data_loader, valid_data_loader

    def train(self):
        current_step = 1
        start = time.time()
        writer = SummaryWriter(log_dir="summary_pic")
        for epoch in range(self.epochs):
            print("----- Epoch {}/{} -----".format(epoch + 1, self.epochs))
            for batch_data in self.train_data_loader:
                input_ids = batch_data[0].cuda()
                attention_mask = batch_data[1].cuda()
                labels = batch_data[2].cuda()
                with autocast():
                    loss = self.model(input_ids, attention_mask, labels)
                if current_step % 100 == 0:
                    writer.add_scalar('loss', loss.detach(), current_step)
                    print(loss)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                current_step += 1

        end = time.time()
        print("total train time: ", end - start)
        output_model_file = os.path.join(self.output, WEIGHTS_NAME)
        output_config_file = os.path.join(self.output, CONFIG_NAME)
        # 保存信息和模型权重
        torch.save(self.model.state_dict(), output_model_file)
        self.model.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(self.output)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()