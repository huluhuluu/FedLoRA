# python -u main_fed.py --model bert-base-uncased --dataset agnews --dirichlet_noniid --num_users 1000 --local_bs 32 --local_ep 1 --lr 0.04 --interval 1 --epochs 300 --gpu 4 --init kaming --alpha 0.01

# python -u main_fed.py --model bert-base-uncased --dataset agnews --dirichlet_noniid --num_users 1000 --local_bs 32 --local_ep 1 --lr 0.04 --interval 1 --epochs 300 --gpu 4 --init loraga --alpha 0.01

python -u main_fed.py --model bert-base-uncased --dataset agnews --dirichlet_noniid --num_users 1000 --local_bs 32 --local_ep 1 --lr 0.1 --interval 1 --epochs 300 --gpu 4 --init pissa --alpha 10.0