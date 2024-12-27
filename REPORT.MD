## 1. Understanding and Utilizing Transformers 

### 1.1. Transformer models : Mathematical foundation and Architecture

Self attention

Positional Encoding 

Feedforward NN

Encoder-Decoder Structure

Scalability

Ref
- attention is all you need
- HF transformers doc + TP 
- Jay Allammar's Visual Guide to Transformers 
- Peter Bloem : transformers from scratch 

### 1.2. Causal Transformers : Concept and Training Methodologies

Causal Masking

Training Objective

Architectures : GPT and LLaMA -> focus on GPT

Challenges and solutions 

Optimization Techniques 


## 2. Reinforcement Learning with Human Feedback 

### 2.1. How does it works ? 

Overview of RLHF

Key steps

Applications 

Advantages and Challenges 

Specific context of LLMs

### 2.2. Training a Reward Model : implementation in practice 

Documentation and analysis of the training process 
https://huggingface.co/docs/trl/dataset_formats#preference -> choice of the dataset

RewardTrainer : Preference (implicit prompt recommended)

The [RewardTrainer] requires a implicit prompt preference dataset. It means that the dataset should only contain the columns "chosen" and "rejected" (and not "prompt"). The [RewardTrainer] supports both conversational and standard dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

You can also use a pretokenized dataset, in which case the dataset should contain the following columns: input_ids_chosen, attention_mask_chosen, input_ids_rejected and attention_mask_rejected.v

After preparing your dataset, you can use the [RewardTrainer] in the same way as the Trainer class from 🤗 Transformers. You should pass an AutoModelForSequenceClassification model to the [RewardTrainer], along with a [RewardConfig] which configures the hyperparameters of the training.

https://github.com/huggingface/trl/tree/main

Dataset Collection 

Reward Model architecture 

Training process (loss functions used, evaluation metrics...)

Importance of robust evaluation 

Common pitfalls like overfitting to the reward model 

conclusion : can be improved / optimized : link to PPO


## 3. Optimization with Proximal Policy Optimization (PPO)

What is PPO (quickly)?

Generation of the sample output from the optimized model to demonstrate its capabilities

PPO FAQ Training :

Mean Reward: The primary goal is to maximize the reward achieved by the model during RL training. Objective KL Divergence: KL divergence 
(Kullback-Leibler divergence) measures the dissimilarity between two probability distributions. In the context of RL training, we use it 
to quantify the difference between the current model and a reference model. Ideally, we want to keep the KL divergence between 0 and 10 to 
ensure the model’s generated text remains close to what the reference model produces.

