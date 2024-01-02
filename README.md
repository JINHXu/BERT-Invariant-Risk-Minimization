## BERT implemented with Invariant Risk Minimization 

(code is trimmed down from https://github.com/technion-cs-nlp/irm-for-nli)

`dataet_utils.py` allows injecting artificial bias symbolized by special tokens. The default setting will not inject any synthetic noise.

#### run irm

`python main.py run-irm --out-dir models/main_exp/irm/run0 --bs-train 32 --bs-val 32 --eval-every-x-epoch 0.2 --warm-up-epochs 1 --epochs 4 --warm-up-reg 1 --reg 7500 --lr 5e-5  --early-stopping 5 --seed 666 --gradient-checkpoint`

#### test irm

`! python main.py test-irm /content/test_ood.txt /content/models/main_exp/irm/run0 --out-dir models/main_exp/irm/run0/test_ood --reg 7500`

adjust hyperparameters and IRM penalty weight as needed.

set `--reg 7500` to run erm
