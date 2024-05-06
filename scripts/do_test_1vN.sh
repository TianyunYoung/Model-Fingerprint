pretrain_model_path=./weights/fingerprint_extractor.pth

python3 test_1vn.py \
--device cuda:3 \
--pretrain_model_path $pretrain_model_path \
--test_data_paths ./dataset/data_list/subset_data.txt \
--window_slide 





