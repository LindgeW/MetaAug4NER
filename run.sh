## baseline
nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 16 --genre conll_5 --aug_genre conll_5_repl2 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl &> conll_5.log &
## mixup
# nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 16 --genre conll_5 --aug_genre conll_5_repl2 --train_type aug --to_mix --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl &> conll_5.log &
## ts + rw
# nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 16 --genre conll_5 --aug_genre conll_5_repl2 --train_type aug --to_rw --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl &> conll_5.log &
## mixup + rw
# nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 16 --genre conll_5 --aug_genre conll_5_repl2 --train_type aug --to_mix --to_rw --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl &> conll_5.log &

# nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 16 --genre onto_5 --aug_genre onto_5_repl1 --train_type aug --model_chkp onto_5_model.pkl --vocab_chkp onto_5_vocab.pkl &> onto_5.log &
# nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 32 --genre onto_5 --aug_genre onto_5_repl2 --train_type aug --model_chkp onto_5_model.pkl --vocab_chkp onto_5_vocab.pkl &> onto_5.log &
# nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 64 --genre onto_5 --aug_genre onto_5_repl3 --train_type aug --model_chkp onto_5_model.pkl --vocab_chkp onto_5_vocab.pkl &> onto_5.log &
# nohup python train.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 16 --aug_batch_size 32 --genre wb --aug_genre wb_repl1 --train_type aug --model_chkp wb_model.pkl --vocab_chkp wb_vocab.pkl &> wb_both.log &
