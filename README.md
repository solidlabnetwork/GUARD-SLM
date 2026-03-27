# GUARD-SLM
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
