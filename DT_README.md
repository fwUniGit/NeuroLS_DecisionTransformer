# NeuroLS Decisionn Transformer Documentation
## Setting up a new dataset to train Decision Transformer
<ol>
<li> Create dataset of jssp instances with _create_jssp_val_test_sets.ipynb_ for respective problem size </li>
<li> Set *env:* in jssp_config.yaml to respective problem size</li>
<li> Define values in eval_jssp.yaml (Uncomment repsective _checkpoint:_ run_type: test,operator_mode:, num_steps set and test_dataset )</li>
<li>run\_nls\_jssp.py</li>
<li>Process with Notebook _create_dt_dataset.ipynb_ to calculate returns-to-go and convert format of dataset</li>
</ol>
