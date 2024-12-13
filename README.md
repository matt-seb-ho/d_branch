# cse291a00 project: ongoing RM experiments
*repo is forked from [OpenR](https://github.com/openreasoner/openr)*

As discussed in our final presentation, our future work includes investigating the impact of branching diversity on reward models in addition to the policy/generator model.

This repo modifies the standard temperature sampling regime to enforce diverse branches in initial data generation used as part of the PRM training pipeline.

## Usage Instructions
#### Environment Setup
```bash
conda create -n open_reasoner python=3.10
conda activate open_reasoner
pip install -r requirements.txt
pip3 install  "fschat[model_worker,webui]"
pip install -U pydantic
cd envs/MATH/latex2sympy
pip install -e .
cd -
```
#### PRM Training Data Synthesis
```bash
cd data/
python gen_diverse_data.py
```
#### PRM Training and Evaluation
- follow instructions from OpenR documentation `https://openreasoner.github.io/docs/usage/prm.html`
