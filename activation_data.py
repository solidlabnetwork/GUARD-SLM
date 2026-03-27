import os
import re
import ast
import json
import argparse
from datetime import datetime

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--max-input-tokens", type=int, default=512)

    p.add_argument("--use-advbench", action="store_true")
    p.add_argument("--use-alpaca", action="store_true")
    p.add_argument("--use-jbkv", action="store_true")

    p.add_argument("--benign-json", action="append", default=[])
    p.add_argument("--malicious-json", action="append", default=[])
    p.add_argument("--other-malicious-json", action="append", default=[])
    p.add_argument("--jailbreak-json", action="append", default=[])

    return p.parse_args()


def sanitize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    return text.strip("_") or "unknown"


def build_prompt(tok, text):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {text}\nAssistant:"


def infer_attack(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    return sanitize(stem.split("_")[0])


def load_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Empty file: {path}")

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            if "results" in data and isinstance(data["results"], list):
                return data["results"]
            return [data]
    except Exception:
        pass

    rows = []
    ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            ok = False
            break
    if ok and rows:
        return rows

    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    raise ValueError(f"Cannot parse {path}")


def get_text(row):
    for k in [
        "jailbreak",
        "jailbreak_prompt",
        "final_query",
        "attack_prompt",
        "rewritten_query",
        "prompt",
        "query",
        "instruction",
        "goal",
        "request",
    ]:
        v = row.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


@torch.no_grad()
def main():
    args = parse_args()

    if not any([
        args.use_advbench,
        args.use_alpaca,
        args.use_jbkv,
        len(args.benign_json) > 0,
        len(args.malicious_json) > 0,
        len(args.other_malicious_json) > 0,
        len(args.jailbreak_json) > 0,
    ]):
        raise ValueError("No dataset provided")

    os.makedirs(args.outdir, exist_ok=True)

    out_path = os.path.join(
        args.outdir,
        f"{args.model.replace('/','-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    print("[+] Loading model...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    L = model.config.num_hidden_layers

    def extract(text):
        prompt = build_prompt(tok, text)
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)

        out = model(**inputs, output_hidden_states=True, use_cache=False)

        for i in range(1, L + 1):
            h = out.hidden_states[i][:, -1, :].squeeze(0)
            yield i - 1, h

    gid = 0
    with open(out_path, "w", encoding="utf-8") as f:

        if args.use_advbench:
            print("[+] Loading AdvBench...")
            ds = load_dataset("Lemhf14/EasyJailbreak_Datasets", "AdvBench", split="train")
            for row in tqdm(ds, desc="AdvBench"):
                text = row.get("query", "")
                if not isinstance(text, str) or not text.strip():
                    continue

                for layer, h in extract(text):
                    f.write(json.dumps({
                        "id": gid,
                        "type": "malicious",
                        "source": "AdvBench",
                        "layer": layer,
                        "activation": h.float().cpu().tolist(),
                    }) + "\n")
                gid += 1

        if args.use_alpaca:
            print("[+] Loading Alpaca...")
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            for row in tqdm(ds, desc="Alpaca"):
                text = row.get("instruction", "")
                if not isinstance(text, str) or not text.strip():
                    continue

                for layer, h in extract(text):
                    f.write(json.dumps({
                        "id": gid,
                        "type": "benign",
                        "source": "alpaca",
                        "layer": layer,
                        "activation": h.float().cpu().tolist(),
                    }) + "\n")
                gid += 1

        if args.use_jbkv:
            print("[+] Loading JBKV...")
            ds = load_dataset(
                "JailbreakV-28K/JailBreakV-28k",
                "JailBreakV_28K",
                split="JailBreakV_28K",
            )
            for row in tqdm(ds, desc="JBKV"):
                text = row.get("jailbreak_query", "")
                if not isinstance(text, str) or not text.strip():
                    continue

                for layer, h in extract(text):
                    f.write(json.dumps({
                        "id": gid,
                        "type": "JBKV",
                        "source": "JBKV",
                        "layer": layer,
                        "activation": h.float().cpu().tolist(),
                    }) + "\n")
                gid += 1

        def run_json_file(path, typ, src):
            nonlocal gid
            print(f"[+] Loading {src}: {path}")
            data = load_json_any(path)

            for row in tqdm(data, desc=src):
                if not isinstance(row, dict):
                    continue

                text = get_text(row)
                if not text:
                    continue

                for layer, h in extract(text):
                    f.write(json.dumps({
                        "id": gid,
                        "type": typ,
                        "source": src,
                        "layer": layer,
                        "activation": h.float().cpu().tolist(),
                    }) + "\n")
                gid += 1

        for path in args.benign_json:
            run_json_file(path, "benign", "benign_json")

        for path in args.malicious_json:
            run_json_file(path, "malicious", "malicious_json")

        for path in args.other_malicious_json:
            run_json_file(path, "malicious", "other_malicious_json")

        for path in args.jailbreak_json:
            cat = infer_attack(path)
            run_json_file(path, cat, f"{cat}_json")

    print(f"\n[✓] Done → {out_path}")


if __name__ == "__main__":
    main()