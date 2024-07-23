# Gemma 2 Fine-Tuning - Reward Model - (Kaggle Competition)

[![Static Badge](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=Kaggle&labelColor=white)](https://www.kaggle.com/)
[![Static Badge](https://img.shields.io/badge/Google-Gemma%202-yellow?logo=google&logoColor=red&labelColor=white)](https://huggingface.co/google/gemma-2-9b)
[![Static Badge](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?logo=huggingface&logoColor=yellow&labelColor=white)](https://huggingface.co/docs/transformers/index)
[![Static Badge](https://img.shields.io/badge/License-MIT-brown?labelColor=white)](https://en.wikipedia.org/wiki/MIT_License)

This project will fine-tune Gemma 2 model for the reward model. That model will inference which response has the better results in pleasing people.

# Overview
In the concept of Reinforcement Learning in LLMs, Reward Model plays an important role. Instead of waiting for human responses about which LLM response is better, we use a reward model to evaluate responses and return reward scores for LLM model to learn. This will improve the triple H problems: honest, harmless and helpful.

![Reinforcement Learning LLM](https://cdn.prod.website-files.com/63f3993d10c2a062a4c9f13c/64c7d68df4bd1ec49d4f0077_1*cNrEpldTOy510JguQBPriQ.png)

To train a reward model, there are several ways using Statistical, Machine Learning, Deep Learning,... In this project, we will fine-tune a model from Google - Gemma 2 using dataset of Human response in a Kaggle Competition. 
