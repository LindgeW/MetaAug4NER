#python train_reweight.py --cuda 2 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 16 --test_batch_size 32 --update_step 1 --patient 3 --genre onto_5 --aug_genre onto_5_repl1 --train_type aug &> onto_5.log2 &

#python train_reweight.py --cuda 6 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 6 --test_batch_size 16 --update_step 1 --patient 3 --genre onto_5 --aug_genre onto_5_repl1 --train_type aug --model_chkp model.pkl --vocab_chkp vocab.pkl &> onto_5_mix.log &

#python train_reweight.py --cuda 2 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 16 --update_step 1 --patient 3 --genre onto_10 --aug_genre onto_10_repl1 --train_type aug --model_chkp model_10.pkl --vocab_chkp vocab_10.pkl &> onto_10_both.log &

#python train_reweight.py --cuda 5 -lr 1e-3 --bert_lr 2e-5 --batch_size 3 --aug_batch_size 6 --test_batch_size 16 --update_step 1 --patient 3 --genre onto_10 --aug_genre onto_10_repl1 --train_type aug --model_chkp model_10.pkl2 --vocab_chkp vocab_10.pkl2 &> onto_10_both.log2 &

#python train_reweight.py --cuda 3 -lr 1e-3 --bert_lr 2e-5 --batch_size 3 --aug_batch_size 6 --test_batch_size 16 --update_step 1 --patient 3 --genre onto_30 --aug_genre onto_30_repl1 --train_type aug --model_chkp model_30.pkl --vocab_chkp vocab_30.pkl &> onto_30_mix.log2 &

#python train_reweight.py --cuda 6 -lr 1e-3 --bert_lr 2e-5 --batch_size 3 --aug_batch_size 6 --test_batch_size 16 --update_step 1 --patient 3 --genre onto_30 --aug_genre onto_30_repl1 --train_type aug --model_chkp model_30.pkl3 --vocab_chkp vocab_30.pkl3 &> onto_30_both.log &


#python train_reweight.py --cuda 3 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 16 --update_step 1 --patient 3 --genre wb --aug_genre wb_repl1 --train_type aug --model_chkp wb_model.pkl --vocab_chkp wb_vocab.pkl &> wb_both.log &

#python train_reweight.py --cuda 4 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre wb --aug_genre wb_repl2 --train_type aug --model_chkp wb_model2.pkl --vocab_chkp wb_vocab2.pkl &> wb_both.log2 &

#python train_reweight.py --cuda 5 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 16 --update_step 1 --patient 3 --genre wb --aug_genre wb_repl1 --train_type aug --model_chkp wb_model3.pkl --vocab_chkp wb_vocab3.pkl &> wb_both.log3 &

#python train_reweight.py --cuda 6 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 16 --update_step 1 --patient 3 --genre wb --aug_genre wb_repl1 --train_type aug --model_chkp wb_model4.pkl --vocab_chkp wb_vocab4.pkl &> wb_both.log4 &

#python train_reweight.py --cuda 2 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 3 --bert_grad_clip 0.5 &> conll_5_both.log2_3 &

#python train_reweight.py --cuda 3 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl2 --train_type aug --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_both.log3_3 &



#python train_reweight.py --cuda 5 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --mix_alpha 0.5 --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_30_mix.log_05 &

#python train_reweight.py --cuda 2 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --mix_alpha 1 --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_30_mix.log_1 &

#python train_reweight.py --cuda 3 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --mix_alpha 3 --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_30_mix.log_3 &

#python train_reweight.py --cuda 4 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --mix_alpha 5 --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_30_mix.log_5 &

#python train_reweight.py --cuda 5 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --mix_alpha 7 --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_30_mix.log_7 &

#python train_reweight.py --cuda 7 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --mix_alpha 9 --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_30_mix.log_9 &

#python train_reweight.py --cuda 7 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 12 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --mix_alpha 11 --model_chkp conll_5_model2.pkl --vocab_chkp conll_5_vocab2.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_30_mix.log_11 &



python train_reweight.py --cuda 1 -lr 1e-3 --bert_lr 2e-5 --batch_size 8 --aug_batch_size 16 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_TS.log &


#python train_reweight.py --cuda 0 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1_0 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_ts0.log &

#python train_reweight.py --cuda 1 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1_2 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_ts2.log &

#python train_reweight.py --cuda 2 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1_4 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_ts4.log &

#python train_reweight.py --cuda 3 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1_6 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_ts6.log &

#python train_reweight.py --cuda 4 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1_8 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_ts8.log &

#python train_reweight.py --cuda 5 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl1_10 --train_type aug --model_chkp conll_5_model.pkl --vocab_chkp conll_5_vocab.pkl --grad_clip 5 --bert_grad_clip 1 &> conll_5_ts10.log &



#python train_reweight.py --cuda 7 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 16 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_5 --aug_genre conll_5_repl3 --train_type aug --model_chkp conll_5_model3.pkl --vocab_chkp conll_5_vocab3.pkl &> conll_5_both.log3 &

#python train_reweight.py --cuda 3 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 32 --update_step 1 --patient 3 --genre conll_10 --aug_genre conll_10_repl1 --train_type aug --model_chkp conll_10_model.pkl --vocab_chkp conll_10_vocab.pkl &> conll_10_both.log2 &

#python train_reweight.py --cuda 5 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 16 --update_step 1 --patient 3 --genre conll_30 --aug_genre conll_30_repl1 --train_type aug --model_chkp conll_30_model.pkl --vocab_chkp conll_30_vocab.pkl &> conll_30_both.log2 &

#python train_reweight.py --cuda 1 -lr 1e-3 --bert_lr 2e-5 --batch_size 4 --aug_batch_size 8 --test_batch_size 16 --update_step 1 --patient 3 --genre conll --aug_genre conll_repl1 --train_type aug --model_chkp conll_model.pkl --vocab_chkp conll_vocab.pkl &> conll_both.log2 &
