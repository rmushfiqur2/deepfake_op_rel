### running the training algorithm

```Shell
cd flr
conda activate ICT_DeepFake
python main-transfer-learning.py

cd flr/datasets
python deepfake_training_data_generator.py
```

### view the logs
``tensorboard --logdir logs``


