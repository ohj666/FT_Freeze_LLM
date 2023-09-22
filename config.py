class Config:
    epochs = 4

    train_steps = 1500
    warmup_steps = 100

    batch_size = 1
    sequence_len = 256
    learning_rate = 3e-4
    weight_decay = 0

    load_in_8bit = False
    val_set_size = 2000
    data_path = "./trans_chinese_alpaca_data.json"
    base_model = "4bit/pyg-7b"