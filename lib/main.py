
import pickle
import numpy as np
from mingpt import dt_trainer, dt_model, dt_cassandra_trainer
from utils.StateActionReturnDataset import StateActionReturnDataset

# ___ CONSTANTS ___
CONTEXT_LENGTH = 50

if __name__ == '__main__':
  # List[List[Task]]
    np.random.seed(123)

    with open("projects/NeuroLS_DecisionTransformer/lib/50x20_100it_1000inst_flat.pkl", "rb") as handle:
        train_data = pickle.load(handle)

    train_dataset = StateActionReturnDataset(train_data, CONTEXT_LENGTH)

    mconf = dt_model.GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8, n_embd=128, max_timestep=199, observation_size=128)
    model = dt_model.GPT(mconf)
    #model.load_state_dict(T.load("balmy-wood-104.pt", map_location=T.device('cpu')))
    test_data = None

    # initialize a trainer instance and kick off training

    tconf = dt_trainer.TrainerConfig(max_epochs=500, batch_size=128, learning_rate=6e-2,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*CONTEXT_LENGTH*3,
                          num_workers=4, seed=123, max_timestep=199)
    trainer = dt_cassandra_trainer.Trainer(model, train_dataset, test_data, tconf) #NOTE Es wird kein Test Dataset verwendet

    trainer.train()
