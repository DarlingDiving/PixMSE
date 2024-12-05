class DefaultConfig(object):

    train_data_root = '/home/aiuser1/ssd_data/xiaodai/person'
    val_data_root = '/home/aiuser1/ssd_data/xiaodai/val/biggan1'
    test_data_root = '/home/aiuser1/ssd_data/xiaodai/wang'
    #test_data_root = '/home/aiuser1/ssd_data/xiaodai/faceshq'
    load_model = True
    load_model_path_2 = './Daset/checkpoints/'
    save_model = True
    save_model_path = "./Daset/checkpoints/"

    seed = 43
    batch_size = 4 #14
    use_gpu = True
    gpu_id = '0'
    num_workers = 4

    img_size = 299

    max_epoch = 100
    lr = 0.00005
    lr_decay = 0.96
    weight_decay = 0

    # dataset and model configurations
    # check load and save model configurations
    mid_loss_weight = 0.5
    noise_scale = 0

opt = DefaultConfig()
