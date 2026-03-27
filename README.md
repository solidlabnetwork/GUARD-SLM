# GUARD-SLM

## 1. Extract Activations

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

## 2. Activation Visualization (t-SNE)

```bash
python activation_analysis.py \
  --input activation_data/meta-llama-Llama-2-7b-chat-hf_YYYYMMDD_HHMMSS.jsonl \
  --layer 31 \
  --max-samples 52000 \
  --outdir activation_figures
```

---

## 3. Train Binary Classifier (Single Layer)

```bash
python activation_classification.py \
  --input activation_data/meta-llama-Llama-2-7b-chat-hf_YYYYMMDD_HHMMSS.jsonl \
  --layer 18 \
  --outdir saved_models/llama \
  --model-name llama_layer_18 \
  --overwrite
```

---

## 4. Train Classifiers for All Layers

```bash
for layer in $(seq 0 31); do
  python activation_classification.py \
    --input activation_data/meta-llama-Llama-2-7b-chat-hf_YYYYMMDD_HHMMSS.jsonl \
    --layer ${layer} \
    --outdir saved_models/llama \
    --model-name llama_layer_${layer} \
    --overwrite
done
```

---

## 5. Inference on HarmBench

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
