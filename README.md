# NeuroLS Decision Transformer Documentation
### Setup
**Note:** For me using Windows the NeuroLS environment is only working when using Windows Subsytem for Linux  (WSL)  

Install the requirements as conda environment
```sh
conda env create -f requirements.yml
```
### Basic Usage
Run Decision Transformer 15x15 model on Taillard Benchmark (For other problem change respective entries in command):
In _eval_jssp.yaml_ the NLS model for the respective problem size must be uncommented.
Further flags are shown in run_benchmark.py
```sh
python run_benchmark.py -r run_nls_jssp.py -d  data/JSSP/benchmark/TA -g 15x15 -p jssp -m nls -e eval_jssp --args "env=jssp15x15_unf" -n 200 -x  15x15/.pt -a True --rtg_factor 1.0 
```
Run NLS 15x15 model on Taillard Benchmark
```sh
python run_benchmark.py -r run_nls_jssp.py -d  data/JSSP/benchmark/TA -g 15x15 -p jssp -m nls -e eval_jssp --args "env=jssp15x15_unf" -n 200
```

#### run_benchmark.py
Handles to run benchmark or test dataset with the NLS or NLSDT (depending if -a flag is True)  
**Important:** _tester_cfg.test_dataset_size= in line 112 must be 1 for Taillard benchmark and 100 (or other testdataset size) otherwise.
Watch out to add a space after the number in the string.
### Setting up a new dataset to train Neuro-LS Decision Transformer
<ol>
    <li> Create dataset of jssp instances with _create_jssp_val_test_sets.ipynb_ for respective problem size 
    </li>
<li> Set **env**: in jssp_config.yaml to respective problem size</li>
<li> Define values in eval_jssp.yaml 
    Make sure to set:
    <ul>
        <li>run_type: must allways be test (We do not want to train the original NLS Model)</li>
        <li>(Uncomment repsective _checkpoint:_ </li>
        <li>test_dataset_size: Size of the dataset that is used for NLS-DT data creation.</li>
        <li>test_batch_size: Must be 1 for NLS-DT</li>
        <li>Operator mode: SET(NLS_A), SELECT_LS(NLS_AN), SELECT_LS+(NLS_ANP)</li>
        <li>num_steps: Number of iterations to run the local search per instance</li>
        <li>data_file_path: Path to instance file created with _create_jssp_val_test_sets notebook</li>
    </ul>
</li>
<li>**run\_nls\_jssp.py** : This will run the NeuroLS model over the test_instances eval_jssp.yaml.
</li>
<li>Process the created data points with _create_dt_dataset.ipynb_ to calculate returns-to-go and convert format of dataset</li>
</ol>

### Train NeuroLS Decision Transformer ###
#### Traindata
The train data is only available on the DVD-ROM appended to the thesis because it exeeds the github file size limit.

run lib/main.py


## Most relevant files
### minGPT ### 
 Contains the code of the original [Decision Transformer](https://github.com/kzl/decision-transformer/ "Named link title") which is based on [minGPT](https://github.com/karpathy/minGPT).
#### dt_model.py #### 
Contains the minGPT pytorch Model adjusted for the D-DT
forward path of the model expects _sequence_ of states, actions, targets rtgs,timestep, action_masks
Note that minGPT refers to the context_length as _block_size_.

**Class: dt_model.GPTConfig(vocab_size, block_size, kwargs)**
vocab_size -> amount of actions in action space
block_size -> context_length  
Furthermore defines via kwards ->  
n_layer -> number of transformer blocks to use
n_head -> number of self-attention heads  
n_embd -> embedding size  
max_timestep -> Max length of RL environment  
observation_size

#### dt_trainer.py ####
This file defines the minGPT training framework

Contains: 

**TrainerConfig**:

Set various parameters like learning rate, batch size, and optimization settings. It includes settings for learning rate decay.
**Trainer**: 

The main class responsible for training the model. Implements a training loop with epoch and batch processing, and includes methods for saving checkpoints and testing the model. Logsprogress to Weights & Biases (wandb).

#### utils.py ####
Implements the funcitonality to pass the sequences to model 
and converts logits returned from the model to the predicted action.


### ./utils ###

#### StateActionReturnDataSet.py ####
class designed for use with PyTorch's Dataset interface.
The state action return dataset is necessary for training. On intialization data in the
format created from _create_data_set_ function is necessary.
__getitem()__ 
returns  for a given index the sequence of the next _context_length_ _observations_, _actions_, rtgs, _timesteps_ 
and time steps. 


#### tianshou_utils ####
<u>_TestCollector.collect()_</u>
Implements the inference of the NLS model and the NLSDT model. In case of DT usage the aggregated state is extracted and passed to the DecisionTransfomerClass  
Extracts datapoints for NLSDT when solving instances and saves data_points.pt to output directory

### DecisionTransformer() ###
Defines the DecisionTransformer class. The class includes methods for calculating return-to-go, updating rewards and states, and sampling actions based on the aggregated environment 

### Experiments ###
To reproducie the exact same experiments without needing to run the whole vary_rtg() process the files experiment_output/nls.zip
and experiment_output/nlsdt.zip need to be unzipped.
The folders contain numpy files for each solved instance named by a unique name per instace in order to compare the instances that are
solved by the NLS and NLSDT. 
The naming is done in: _lib/env/jssp_env.py_

### Important NLS files with additional changes made for NLSDT ### 
**<u>_lib/env/jssp_env.py_:</u>** 

Gym environment to solve Job Shop Scheduling Problems based on Local Search (LS).
Here it was necessary to add the following to the observation:
<ul>
<li>'instance_lower_bound'</li>
<li>'instance_hash'</li>
<li>'machine_sequence'</li>
<li>'starting_times'</li>
</ul>
Also the calc_lower_bound() method is implemented in this environment class

**<u>_lib/scheduling/jssp_graph.py_:</u>** 
Manages the whole disjunctive graph of an instance. Further needed
 to get machine sequence and starting times of instance. ()
