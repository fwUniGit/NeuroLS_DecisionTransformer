
import pickle
import numpy as np
from lib.mingpt import dt_trainer, dt_model
from lib.utils.StateActionReturnDataset import StateActionReturnDataset

# ___ CONSTANTS ___
CONTEXT_LENGTH = 10

if __name__ == '__main__':
  # List[List[Task]]
    np.random.seed(123)

    with open("data_points_nls_flat.pkl", "rb") as handle:
        train_data = pickle.load(handle)

    train_dataset = StateActionReturnDataset(train_data[:1000], CONTEXT_LENGTH)

    mconf = dt_model.GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8, n_embd=128, max_timestep=99, observation_size=128)
    model = dt_model.GPT(mconf)
    #model.load_state_dict(T.load("balmy-wood-104.pt", map_location=T.device('cpu')))
    test_data = None

    # initialize a trainer instance and kick off training

    tconf = dt_trainer.TrainerConfig(max_epochs=10, batch_size=1, learning_rate=6e-4,
                          lr_decay=False, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*CONTEXT_LENGTH,
                          num_workers=4, seed=123, max_timestep=99)
    trainer = dt_trainer.Trainer(model, train_dataset, test_data, tconf) #NOTE Es wird kein Test Dataset verwendet

    trainer.train()
