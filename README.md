# Distributionally robust neural posterior estimation (DRNPE)

Create a virtual environment and install required packages:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run this command from the root directory to train the encoder:
```
nohup python -u drnpe/train.py -cn config.yaml &> output.out &
```

During or after training, check metrics:
```
tensorboard --logdir=logs &
```

See `gaussian_example.ipynb` for an example.
