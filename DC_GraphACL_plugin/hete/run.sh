python main.py --dataset texas --epoch_num 50 --lr 0.0001 --lambda_loss 1.0 --moving_average_decay 0.95 --dimension 2048 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.75
python main.py --dataset wisconsin --epoch_num 50 --lr 0.0001 --lambda_loss 1.0 --moving_average_decay 0.97 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5
python main.py --dataset cornell --epoch_num 50 --lr 0.0001 --lambda_loss 1.0 --moving_average_decay 0.95 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5


