# Gemma 2 Fine-Tuning - Reward Model - (Kaggle Competition)

[![Static Badge](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=Kaggle&labelColor=white)](https://www.kaggle.com/)
[![Static Badge](https://img.shields.io/badge/Google-Gemma%202-yellow?logo=google&logoColor=red&labelColor=white)](https://huggingface.co/google/gemma-2-9b)
[![Static Badge](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?logo=huggingface&logoColor=yellow&labelColor=white)](https://huggingface.co/docs/transformers/index)
[![Static Badge](https://img.shields.io/badge/License-MIT-brown?labelColor=white)](https://en.wikipedia.org/wiki/MIT_License)

This project will fine-tune Gemma 2 model for the reward model. That model will inference which response has the better results in pleasing people.

Gemma 2 is an advanced language model developed by Google, featuring a 9-billion parameter architecture. In this notebook, I use Gemma 2 version 9b-pt.

# Overview

In the concept of Reinforcement Learning in LLMs, Reward Model plays an important role. Instead of waiting for human responses about which LLM response is better, we use a reward model to evaluate responses and return reward scores for LLM model to learn. This will improve the triple H problems: honest, harmless and helpful.

![Reinforcement Learning LLM](https://cdn.prod.website-files.com/63f3993d10c2a062a4c9f13c/64c7d68df4bd1ec49d4f0077_1*cNrEpldTOy510JguQBPriQ.png)

To train a reward model, there are several ways using Statistical, Machine Learning, Deep Learning,... In this project, we will fine-tune a model from Google - Gemma 2 using dataset of Human response in a Kaggle Competition. 

- Data Sample:

| prompt    | response_a | response_b | winner_model_a | winner_model_b | winner_tie |
|-----------|------------|------------|----------------|----------------|------------|
| 1 + 1 = ? | 2          | 3          | 1              | 0              | 0          |
| 2 + 2 = ? | 4          | 5          | 1              | 0              | 0          |
| 3 + 3 = ? | 6          | 6          | 0              | 0              | 1          |

So, we will create a classification model using LLM - Gemma 2.

# Input Template and QLora Config (Quantization + Lora) 

- **Input Template**

```
Prompt: {} 

Response_a: {}

Response_b: {}

--------

Which one is better ?
```

- **Quantization and Lora Config**

This help me a lots! Because of with only personal resources, I can't get working on full fine-tuning with high accuracy (Normal Compute with float32). So it's necessary with 4-bit Quantization and Lora in some target modules with acceptable rank.

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 8,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # Avoid overfitting
    bias = 'none',
    task_type = 'SEQ_CLS'
)
```

# Fine-Tuning Progress

Inital Fine-tuning we will use 500 templates for train and 50 for valid.

- **Resources**
 
```
GPU T4 x2 (Kaggle Supplying)
Time: 18834.2s 
```

- **Trainer config**

```
per_device_train_batch_size= 5
num_train_epochs=2
logging_steps=20
save_steps=20
learning_rate=5e-5
eval_steps=20
evaluation_strategy='steps'
save_total_limit=2
load_best_model_at_end = True
```

- **Initial Fine-tuning Progress**

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 20   | 2.257500      | 2.044791        |
| 40   | 1.654100      | 1.632948        |
| 60   | 1.563700      | 1.599245        |
| 80   | 1.866500      | 1.832489        |
| 100  | 1.663200      | 1.424799        |
| 120  | 1.227800      | 1.574739        |
| 140  | 1.601900      | 2.004847        |
| 160  | 1.237200      | 1.556493        |
| 180  | 1.413400      | 1.370673        |
| 200  | 1.324500      | 1.371178        |

# Result and Inference

- **Result**
Because of the limitation of resources, I have fine-tuned with only 500 for training and 50 for validation. The logloss numbers are still high and   accuracy scores are low. So this project is called "Initial Fine-tuning", we will need to base on this inition and next fine-tuning again and again. 

And I am on the way to next fine-tuning. This project help me to improve my skills on LLMs about Fine-tuning with QLora, Optimize Inference, Classification with LLM,...

- **Example Inference**

```
Prompt: You are a mediator in a heated political debate between two opposing parties. Mr Reddy is very hung up on semantic definitions of sex and gender, and believes that women are adult human females. Meanwhile Ms Blue is extremely fluid with definitions and does not care about truth. He (Ms blue uses he\/him pronouns) insists that anybody can be any gender, gametes don't mean anything, and that men can get pregnant. You, Mr Goddy are tasked with helping them both find a middle ground. 

--------

Response_a: Thank you for sharing the details of the situation. As a mediator, I understand the importance of finding a middle ground that both parties can agree upon.

It is important to recognize that language and definitions can be complex and can vary depending on cultural and societal norms. At the same time, it is also important to acknowledge the biological differences between males and females.

One possible approach to finding a middle ground could be to focus on respecting individuals' self-identification and autonomy while also acknowledging the biological differences between males and females. This could involve using inclusive language that reflects individuals' gender identities while also recognizing the importance of biological sex in certain contexts, such as medical treatment and reproductive health.

It may also be helpful to explore ways to create a more inclusive society where individuals of all gender identities are valued and respected. This could involve promoting education and awareness about gender diversity and creating policies and practices that support individuals' gender identities and expression.

Ultimately, the goal of mediation is to find a solution that meets the needs and concerns of both parties. By focusing on mutual respect, open communication, and a willingness to find common ground, it is possible to reach a resolution that allows both parties to move forward in a positive and productive way. 

Response_b: Mr Reddy and Ms Blue both have valid points in their arguments. On one hand, Mr Reddy is correct that women are adult human females, as defined by their biological and physical characteristics. However, Ms Blue is right that gender is a more fluid concept and that people can identify with any gender they choose. A compromise could be that gender is defined by an individual's self-identification, while sex is defined by their biological characteristics. 

--------

Which one is better ?


Answer: Model a
```

# References

Kaggle. (n.d.). LMSYS Chatbot Arena. Retrieved July 25, 2024, from https://www.kaggle.com/competitions/lmsys-chatbot-arena

Hugging Face. (n.d.). google/gemma-2-9b. Retrieved July 25, 2024, from https://huggingface.co/google/gemma-2-9b

Sabay BioMetzger. (2023, May 17). Fine-tuning GEMMA-2B for binary classification (4-bit quantization). Medium. Retrieved July 25, 2024, from https://medium.com/@sabaybiometzger/fine-tuning-gemma-2b-for-binary-classification-4-bit-quantization-60437e877723
