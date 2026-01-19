# Fine-Tuning-LLaMA-2-7B-using-QLoRA-on-Google-Colab
This project demonstrates how to fine-tune the LLaMA-2 7B chat model using QLoRA on a free Google Colab Tesla T4 GPU. Since large language models require very high GPU memory for full training, this project uses 4-bit quantization and LoRA adapters to reduce memory usage. This makes it possible to train a large model even on limited hardware.

The base model used is NousResearch/Llama-2-7b-chat-hf and the training dataset is Databricks Dolly-15k, which contains instruction–response examples. The dataset is reformatted into a structured prompt style so the model can learn better instruction-following behavior.

QLoRA is applied by loading the model in 4-bit precision and training only small LoRA adapter layers instead of all model parameters. This approach saves memory and speeds up training while maintaining good performance. Training is done using the TRL library’s SFTTrainer.

Because the Tesla T4 GPU does not support BF16 precision, mixed precision training is disabled to avoid runtime errors. The model is trained for one epoch with a small batch size and gradient accumulation to stay within memory limits.

This project shows how large language models can be fine-tuned efficiently using modern optimization techniques, even without access to high-end GPUs.
