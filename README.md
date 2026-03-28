# GUARD-SLM: Token Activation-Based Defense Against Jailbreak Attacks for Small Language Models

GUARD-SLM is a lightweight, inference-time jailbreak defense that detects malicious prompts using **last-token hidden-layer activations** without requiring model retraining.

---

## Overview

- Detects jailbreak prompts using **representation-space signals**
- Uses **single forward pass (no extra tokens)**
- Works across multiple jailbreak attack families
- Evaluated on both **SLMs and LLMs**

---

## Installation

```bash
git clone https://github.com/solidlabnetwork/GUARD-SLM.git
cd GUARD-SLM

pip install -r requirements.txt
```

---

## Models

### Small Language Models (SLMs)

| Name         | Model ID                              | Link |
|--------------|--------------------------------------|------|
| LLaMA-2-7B   | meta-llama/Llama-2-7b-chat-hf        | https://huggingface.co/meta-llama/Llama-2-7b-chat-hf |
| Vicuna-7B    | lmsys/vicuna-7b-v1.5                 | https://huggingface.co/lmsys/vicuna-7b-v1.5 |
| Mistral-7B   | mistralai/Mistral-7B-Instruct-v0.2   | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 |
| Yi-6B        | 01-ai/Yi-6B-Chat                     | https://huggingface.co/01-ai/Yi-6B-Chat |
| Qwen-7B      | Qwen/Qwen1.5-7B-Chat                 | https://huggingface.co/Qwen/Qwen1.5-7B-Chat |
| Gemma-7B     | google/gemma-7b-it                   | https://huggingface.co/google/gemma-7b-it |
| OpenChat-3.5 | openchat/openchat-3.5-0106           | https://huggingface.co/openchat/openchat-3.5-0106 |

---

### Large Language Models (LLMs)

| Name         | Model ID                              | Link |
|--------------|--------------------------------------|------|
| LLaMA-2-13B  | meta-llama/Llama-2-13b-chat-hf       | https://huggingface.co/meta-llama/Llama-2-13b-chat-hf |
| Vicuna-13B   | lmsys/vicuna-13b-v1.5                | https://huggingface.co/lmsys/vicuna-13b-v1.5 |
| Qwen-14B     | Qwen/Qwen1.5-14B-Chat                | https://huggingface.co/Qwen/Qwen1.5-14B-Chat |

---

## Datasets

| Dataset Name           | Hugging Face ID                         | Link |
|-----------------------|----------------------------------------|------|
| AdvBench              | Lemhf14/EasyJailbreak_Datasets         | https://huggingface.co/datasets/Lemhf14/EasyJailbreak_Datasets |
| Alpaca                | tatsu-lab/alpaca                       | https://huggingface.co/datasets/tatsu-lab/alpaca |
| JailbreakV-28K (JBKV) | JailbreakV-28K/JailBreakV-28k          | https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k |
| HarmBench             | thu-coai/AISafetyLab_Datasets          | https://huggingface.co/datasets/thu-coai/AISafetyLab_Datasets |

---

## Additional Malicious Data

- Located in:
  ```
  Dataset/Train/llama/v2/
  ```

- Includes diverse jailbreak attacks:
  - AutoDAN
  - GCG
  - PAIR
  - Cipher
  - DeepInception
  - CodeChameleon
  - ICA
  - Jailbroken
  - TAP

- Some malicious data collected from:
  - JBShield (https://github.com/NISPLab/JBShield/tree/main/data)

---

## Experiments

### 1. Extract Activations

```bash
python main.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --use-advbench \
  --use-alpaca \
  --other-malicious-json data/GUARD-SLM/Dataset/Train/llama/v2/malicious.json \
  --jailbreak-json data/Train/llama/v1/autodan_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v2/autodan_llama-2.json \
  --jailbreak-json data/Train/llama/v1/cipher_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v1/codechamelon_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v1/deepinception_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v1/gcg_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v2/gcg_llama-2.json \
  --jailbreak-json data/Train/llama/v1/ica_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v1/jailbroken_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v1/pair_advbench_llama-2_7B.jsonl \
  --jailbreak-json data/Train/llama/v2/pair_llama-2.json \
  --jailbreak-json data/Train/llama/v1/tap_advbench_llama-2_7B.jsonl \
  --outdir activation_data
```

---

### 2. Visualization (t-SNE)

```bash
python activation_analysis.py \
  --input activation_data/<generated_file>.jsonl \
  --layer 31 \
  --max-samples 52000 \
  --outdir activation_figures
```

---

### 3. Train Classifier (Single Layer)

```bash
python activation_classification.py \
  --input activation_data/<generated_file>.jsonl \
  --layer 18 \
  --outdir saved_models/llama \
  --model-name llama_layer_18 \
  --overwrite
```

---

### 4. Train for All Layers

```bash
for layer in $(seq 0 31); do
  python activation_classification.py \
    --input activation_data/<generated_file>.jsonl \
    --layer ${layer} \
    --outdir saved_models/llama \
    --model-name llama_layer_${layer} \
    --overwrite
done
```

---

### 5. Inference (HarmBench)

```bash
for layer in $(seq 0 31); do
  python inference.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --svm-path saved_models/llama/llama_layer_${layer}.joblib \
    --layer ${layer} \
    --input-file thu-coai/AISafetyLab_Datasets \
    --hf-subset harmbench \
    --hf-split standard \
    --text-key query \
    --out-dir outputs/llama_harmbench
done
```

---

## 📈 Evaluation

- Metric: **Attack Success Rate (ASR)**
- Evaluator: **GPT-4o and GPT-4o-mini**

---
### 5. Judge

```bash
export OPENAI_API_KEY="your api key"

python judge.py \
  --input-file  your path \
  --output-dir judge_results
```

---

## Citation

```bibtex

```

---
