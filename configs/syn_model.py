class Config(object): 
    # random seed
    seed = 0

    # ------ dataset ------
    img_sources = ['celeba', 'lsun_objects']
    batch_size = 200
    num_workers = 8

    # ------ training classifier ------
    init_lr = 1e-4
    max_epochs = 40
    margin = 0.3
    T_0 = 15
    T_mult = 2
    eta_min = 1e-5
    w_cls = 1
    w_metric = 1
    save_interval = 10

    # ------ synthesized models ------
    syn_model_dir='./weights/syn_models/'
    model_num = 20
    arch_id_list = []
    for id_1 in range(2):
        for id_2 in range(2):
            for id_3 in range(2):
                for id_4 in range(3):
                    for id_5 in range(4):
                        for id_6 in range(3):
                            componet_id_list = f'{id_1}{id_2}{id_3}{id_4}{id_5}{id_6}'
                            arch_id_list.append(componet_id_list)
