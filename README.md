# Xformer - Framework for evaluation of NLP tasks


## Usage 
---
<br>


### 1. Using config.json
 See `summ.json` or `eval.json` for execution configuration and `dsconfig_zero2.json` for deepspeed configuration
```bash
torchrun main.py --config <path_to_json> --deepspeed <path_to_json>
```


### 2. Deploying on the LRZ AI Systems
Deployment script suports `--help` option.

```bash
python lrz/deploy.py --exec summ --gpu lrz-dgx-a100-80x8 --num_gpus 2 --max_time 4000 --dsconfig 2
```




## Notes

- If using the LRZ AI systems make sure to configure your .ssh/ files accordingly, such that the evaluation results are committed to Gitlab.

- `run_hf_eval.py` is intended to run evaluation on arbitrary HuggingFace Seq2Seq models

- `llm_summ_eval.py` is intended to run evaluation on arbitrary LLMs from HuggingFace (can also be used from within the LRZ script)

- `run_chagpt_eval.ipynb` is intended to run summarization eval using the OpenAI API

## Architecture

### 1.Configuration

Defined in config.py, there are several things happening:

- Reading execution variables (which model, dataset, execute script, ...?) from the --config json
- Creating execute log folders
- Setting environment variables for correct 3rd party code usage
- HuggingFace login and setting the cache folders in the data container: datasets in 
- WandB login for run logging through HF trainer
 


### 2. Datasets


See the ds folder:

- hfdataset.py is the interface class, common functionality to the HF datasets package can be added here
- Currently preprocessing is done automatically as per the SCROLLS procedure since we were only doing summarization. We assume text/summary is contained in the datasets, hfdataset needs reworking to extend to other tasks 
- To extend the datasets: if summarization just add the appropriate variables in the new .py and subclass hfdataset.py


### 3. Models

See the models folder:
- hfmodel.py is the base class which includes tokenizer and model + required functions (tokenize, generate, ...)
- model.py is a interface class in case we wanted to use models besides the hf ones
- extending is trivial if HF model (see bigbirdpegasus.py for var defs)
- for working with LLMs 


### 4. Metrics

Really basic metric interface in metrics/metric.py. Adding addtional metrics should just be creating the file and adding to supported.py


### 5. Execution


Execution scripts to perform different tasks (in exec folder)

- An exec script should have an accompanying json config file with the needed vars (see summ.json or eval.json for finetune.py and eval.py respectively)
- See finetune.py constructor for example usage of the config
- Other usecases besides summarization will have different finetuning and eval work flows, don't just copy the current ones blindly.


### 6. Others

- /utils has saving/reading functions (including pushing to git if .ssh is set)
- Check deepspeed library documentation for anything related to that (shouldn't need to change the config files)