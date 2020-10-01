python src/0_gen_feature_v1.py --train-file public_dat/train.data ^
    --valid-file public_dat/validation.data ^
    --test-file public_dat/test.data ^
    --feat-name-file public_dat/feature.name ^
    --sol-file public_dat/train.solution ^
    --train-feature-file features/train_v1.csv.gz ^
    --valid-feature-file features/valid_v1.csv.gz ^
    --test-feature-file features/test_v1.csv.gz


python src/1_txt_preprocess.py --data-path public_dat --output-file train_val_data_pre.csv.gz

python src/2_txt_gen_ft.py --data-path public_dat ^
    --train-feature-file features/embed_train_v1.csv.gz ^
    --valid-feature-file features/embed_valid_v1.csv.gz ^
    --test-feature-file features/embed_test_v1.csv.gz

python src/2_txt_gen_ft.py --data-path public_dat ^
    --train-feature-file features/embed_train_64_v1.csv.gz ^
    --valid-feature-file features/embed_valid_64_v1.csv.gz ^
    --test-feature-file features/embed_test_64_v1.csv.gz ^
    --embed-size 64

python src/2_txt_gen_ft.py --data-path public_dat ^
    --train-feature-file features/embed_train_128_v1.csv.gz ^
    --valid-feature-file features/embed_valid_128_v1.csv.gz ^
    --test-feature-file features/embed_test_128_v1.csv.gz ^
    --embed-size 128

python src/3_train_predict_lgb.py --

python src/4_txt_entity_preprocess.py --data-path public_dat

python src/5_txt_hashtag_mention_preprocess.py --data-path public_dat


python src/7_gen_feature_v2.py --train-file public_dat/train.data ^
    --valid-file public_dat/validation.data ^
    --test-file public_dat/test.data ^
    --feat-name-file public_dat/feature.name ^
    --sol-file public_dat/train.solution ^
    --train-feature-file features/train_v2.csv.gz ^
    --valid-feature-file features/valid_v2.csv.gz ^
    --test-feature-file features/test_v2.csv.gz


python src/8_gen_sentiment_feature.py --train-file features/train_v1.csv.gz ^
    --valid-file features/valid_v1.csv.gz ^
    --test-file features/test_v1.csv.gz ^
    --train-feature-file features/train_sent.csv.gz ^
    --valid-feature-file features/valid_sent.csv.gz ^
    --test-feature-file features/test_sent.csv.gz


python src/9_gen_feature_v3.py --train-file public_dat/train.data ^
    --valid-file public_dat/validation.data ^
    --test-file public_dat/test.data ^
    --feat-name-file public_dat/feature.name ^
    --sol-file public_dat/train.solution ^
    --train-feature-file features/train_v3.csv.gz ^
    --valid-feature-file features/valid_v3.csv.gz ^
    --test-feature-file features/test_v3.csv.gz

python src/3_entities_agg_ft.py --data-path public_dat ^
    --model-path models ^
    --out-path features ^
    --embed-size 64 ^
    --epoch 10
    
python src/11_lgb_v9.py --feature-file features/all_feat.pickle ^
    --valid-pred-file val_pred/lgb_v9.bag.txt ^
    --test-pred-file pred/lgb_v9.full.bag.seed777.txt ^
    --random-state 777


python src/11_lgb_v9.py --feature-file features/all_train_test.pickle ^
    --valid-pred-file val_pred/lgb_v9.bag.txt ^
    --test-pred-file pred/lgb_v9.full.bag.seed777.txt ^
    --random-state 777


python src/12_lgb_v10.py --feature-file features/all_feat.pickle ^
    --valid-pred-file val_pred/lgb_v10.bag.txt ^
    --test-pred-file pred/lgb_v10.full.bag.seed888.txt ^
    --random-state 888


python src/3_daily_hashtag_agg_ft.py --data-path features ^
    --embed-size 8 ^
    --epoch 10 ^
    --train-feature-file features/train_daily_ht_emb.csv.gz ^
    --valid-feature-file features/valid_daily_ht_emb.csv.gz ^
    --test-feature-file features/test_daily_ht_emb.csv.gz



python src/3_hourly_hashtag_agg_ft.py --data-path features ^
    --embed-size 8 ^
    --epoch 10 ^
    --train-feature-file features/train_hourly_ht_emb.csv.gz ^
    --valid-feature-file features/valid_hourly_ht_emb.csv.gz ^
    --test-feature-file features/test_hourly_ht_emb.csv.gz
