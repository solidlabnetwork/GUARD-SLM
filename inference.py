import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import joblib
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, help="Hugging Face model id")
    p.add_argument("--svm-path", type=str, required=True, help="Path to saved sklearn Pipeline (.joblib)")
    p.add_argument("--layer", type=int, default=31, help="0-based transformer layer index")

    p.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Either a local .json/.jsonl file or a Hugging Face dataset name",
    )
    p.add_argument("--hf-subset", type=str, default="", help="HF dataset subset/config")
    p.add_argument("--hf-split", type=str, default="train", help="HF dataset split")

    p.add_argument("--text-key", type=str, default="query", help="Preferred text field")
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=512)

    p.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    p.add_argument("--out-dir", type=str, required=True, help="Base directory to save outputs")
    p.add_argument("--refusal", type=str, default="I can’t help with that request.")

    p.add_argument(
        "--no-generate-allowed",
        action="store_true",
        help="If set, allowed samples store '<ALLOWED>' instead of actual generation.",
    )

    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )

    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use, e.g. 0 or 1",
    )

    return p.parse_args()


def normalize_subset(subset: Optional[str]) -> str:
    if subset is None:
        return ""
    subset = str(subset).strip()
    if subset in {"", '""', "''", "none", "null", "None", "NULL"}:
        return ""
    return subset


def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\.(json|jsonl)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def get_model_tag(model_name: str) -> str:
    return sanitize_name(model_name.replace("/", "-"))


def is_local_json_path(path: str) -> bool:
    lower = path.lower()
    return os.path.isfile(path) and (lower.endswith(".json") or lower.endswith(".jsonl"))


def get_input_tag(args) -> str:
    if is_local_json_path(args.input_file):
        return sanitize_name(os.path.basename(args.input_file))

    subset = normalize_subset(args.hf_subset)
    if subset:
        return sanitize_name(f"{args.input_file}_{subset}_{args.hf_split}")
    return sanitize_name(f"{args.input_file}_{args.hf_split}")


def get_dataset_name(args) -> str:
    if is_local_json_path(args.input_file):
        return sanitize_name(os.path.basename(args.input_file))

    subset = normalize_subset(args.hf_subset)
    if subset:
        return f"{args.input_file}:{subset}:{args.hf_split}"
    return f"{args.input_file}:{args.hf_split}"


def build_run_tag(args) -> str:
    model_tag = get_model_tag(args.model)
    input_tag = get_input_tag(args)
    return f"{model_tag}_{input_tag}_layer{args.layer}"


def build_output_paths(args):
    run_tag = build_run_tag(args)
    run_dir = os.path.join(args.out_dir, run_tag)
    jsonl_path = os.path.join(run_dir, "results.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")
    return run_tag, run_dir, jsonl_path, summary_path


def build_refusal_rates_csv_path(args) -> str:
    model_tag = get_model_tag(args.model)
    input_tag = get_input_tag(args)
    return os.path.join(args.out_dir, f"{model_tag}_{input_tag}_refusal_rates.csv")


def append_refusal_rate_row(
    csv_path: str,
    layer: int,
    refusal_rate: float,
    refusals: int,
    total_samples: int,
    model: str,
    dataset: str,
    input_source: str,
    run_dir: str,
    results_jsonl: str,
    summary_json: str,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "layer",
        "refusal_rate",
        "refusals",
        "total_samples",
        "model",
        "dataset",
        "input_source",
        "run_dir",
        "results_jsonl",
        "summary_json",
    ]

    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "layer": layer,
                "refusal_rate": round(refusal_rate, 6),
                "refusals": refusals,
                "total_samples": total_samples,
                "model": model,
                "dataset": dataset,
                "input_source": input_source,
                "run_dir": run_dir,
                "results_jsonl": results_jsonl,
                "summary_json": summary_json,
            }
        )


def build_prompt(tokenizer, user_text: str) -> str:
    user_text = "" if user_text is None else str(user_text)

    if getattr(tokenizer, "chat_template", None):
        try:
            messages = [{"role": "user", "content": user_text}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"User: {user_text}\nAssistant:"


def gate_predict_only(clf, x_vec_np) -> int:
    return int(clf.predict(x_vec_np.reshape(1, -1))[0])


def extract_human_from_chosen(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    text = text.strip()
    m = re.search(r"Human:\s*(.*?)(?:\n\s*\nAssistant:|Assistant:|$)", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def load_local_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input JSON/JSONL file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e

                if not isinstance(obj, dict):
                    raise ValueError(f"Line {line_no} is not a JSON object.")
                rows.append(obj)
        return rows

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            if not all(isinstance(x, dict) for x in obj):
                raise ValueError("JSON file must contain a list of objects.")
            return obj

        raise ValueError("JSON file must contain an object or a list of objects.")

    raise ValueError("Unsupported local file type. Use .json or .jsonl")


def load_rows_from_input(args) -> List[Dict[str, Any]]:
    if is_local_json_path(args.input_file):
        return load_local_json_or_jsonl(args.input_file)

    subset = normalize_subset(args.hf_subset)
    if subset:
        ds = load_dataset(args.input_file, subset, split=args.hf_split)
    else:
        ds = load_dataset(args.input_file, split=args.hf_split)

    return [ds[i] for i in range(len(ds))]


def select_user_text(row: Dict[str, Any], preferred_key: str) -> str:
    candidate_keys = [
        preferred_key,
        "query",
        "final_query",
        "jailbreak_prompt",
        "prompt",
        "instruction",
        "input",
        "question",
        "text",
        "chosen",
        "goal",
        "goals",
    ]

    for k in candidate_keys:
        if k in row and row[k] is not None:
            val = row[k]
            val = val if isinstance(val, str) else str(val)

            if k == "chosen":
                return extract_human_from_chosen(val)
            return val

    return ""


def get_used_id(row: Dict[str, Any], idx: int) -> Any:
    for key in ["id", "idx", "index", "uid"]:
        if key in row:
            return row[key]
    return idx


def sanitize_generation_config(model, tokenizer, max_new_tokens: int):
    if hasattr(model, "generation_config") and model.generation_config is not None:
        gc = model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.max_new_tokens = max_new_tokens
        gc.pad_token_id = tokenizer.pad_token_id
        gc.eos_token_id = tokenizer.eos_token_id


@torch.no_grad()
def generate_allowed_text(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
) -> str:
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        temperature=None,
        top_p=None,
    )

    gen_ids = outputs[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def main():
    args = parse_args()
    args.hf_subset = normalize_subset(args.hf_subset)

    if not os.path.exists(args.svm_path):
        raise FileNotFoundError(f"Missing SVM model file: {args.svm_path}")

    run_tag, run_dir, out_jsonl, summary_json = build_output_paths(args)
    refusal_rates_csv = build_refusal_rates_csv_path(args)
    os.makedirs(run_dir, exist_ok=True)

    if torch.cuda.is_available():
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid --gpu {args.gpu}. Available GPU count: {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    rows = load_rows_from_input(args)
    data_mode = "local-json" if is_local_json_path(args.input_file) else "hf-dataset"

    if not rows:
        raise ValueError("No rows found in input.")

    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else len(rows)
    max_samples = min(max_samples, len(rows))

    print(f"Model               : {args.model}")
    print(f"Device              : {device}")
    print(f"Input mode          : {data_mode}")
    print(f"Input source        : {args.input_file}")
    print(f"Dataset             : {get_dataset_name(args)}")
    print(f"Text key            : {args.text_key}")
    print(f"Gate layer          : {args.layer}")
    print(f"Max input tokens    : {args.max_input_tokens}")
    print(f"Max new tokens      : {args.max_new_tokens}")
    print(f"Max samples         : {max_samples}")
    print(f"Run tag             : {run_tag}")
    print(f"Run directory       : {run_dir}")
    print(f"Results JSONL       : {out_jsonl}")
    print(f"Summary JSON        : {summary_json}")
    print(f"Refusal CSV         : {refusal_rates_csv}")
    print(f"Generate allowed    : {not args.no_generate_allowed}")
    print(f"Sampling            : False")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).eval()

    model = model.to(device)
    sanitize_generation_config(model, tokenizer, args.max_new_tokens)

    num_layers = int(getattr(model.config, "num_hidden_layers", -1))
    if num_layers > 0 and not (0 <= args.layer < num_layers):
        raise ValueError(f"Layer {args.layer} invalid. num_hidden_layers={num_layers}")

    clf = joblib.load(args.svm_path)

    total = 0
    refused = 0
    total_inference_time = 0.0
    dataset_name = get_dataset_name(args)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for idx in range(max_samples):
            row = rows[idx]
            used_id = get_used_id(row, idx)
            query = select_user_text(row, args.text_key)
            prompt = build_prompt(tokenizer, query)

            start_time = time.perf_counter()

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model.config.max_position_embeddings,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("hidden_states is None while output_hidden_states=True.")

            target_idx = args.layer + 1
            if not (0 <= target_idx < len(hs)):
                raise ValueError(
                    f"Requested layer {args.layer}, but hidden_states only contains "
                    f"{len(hs) - 1} transformer layers."
                )

            h_last = hs[target_idx][:, -1, :]
            x = h_last.squeeze(0).float().cpu().numpy()

            pred = gate_predict_only(clf, x)
            is_refused = (pred == 1)

            if is_refused:
                generated_text = args.refusal
            else:
                if args.no_generate_allowed:
                    generated_text = "<ALLOWED>"
                else:
                    generated_text = generate_allowed_text(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                        max_new_tokens=args.max_new_tokens,
                    )

            inference_time_sec = time.perf_counter() - start_time
            total_inference_time += inference_time_sec

            total += 1
            if is_refused:
                refused += 1

            out_obj = {
                "used_id": used_id,
                "model": args.model,
                "dataset": dataset_name,
                "query": query,
                "generated_text": generated_text,
                "refused": bool(is_refused),
                "inference_time_sec": round(inference_time_sec, 6),
            }

            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if total % 25 == 0 or total == max_samples:
                rate = refused / total
                print(f"[{total}/{max_samples}] refusals={refused} rate={rate:.4f}")

    rate = refused / max(total, 1)
    avg_inference_time = total_inference_time / max(total, 1)

    summary = {
        "run_tag": run_tag,
        "run_dir": run_dir,
        "results_jsonl": out_jsonl,
        "model": args.model,
        "dataset": dataset_name,
        "input_source": args.input_file,
        "input_mode": data_mode,
        "hf_subset": args.hf_subset,
        "hf_split": args.hf_split,
        "text_key": args.text_key,
        "layer": args.layer,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": max_samples,
        "generated_for_allowed": (not args.no_generate_allowed),
        "refusal_text": args.refusal,
        "total_samples": total,
        "refusals": refused,
        "refusal_rate": round(rate, 6),
        "avg_inference_time_sec": round(avg_inference_time, 6),
        "device": str(device),
        "svm_path": args.svm_path,
        "refusal_rates_csv": refusal_rates_csv,
    }

    with open(summary_json, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    append_refusal_rate_row(
        csv_path=refusal_rates_csv,
        layer=args.layer,
        refusal_rate=rate,
        refusals=refused,
        total_samples=total,
        model=args.model,
        dataset=dataset_name,
        input_source=args.input_file,
        run_dir=run_dir,
        results_jsonl=out_jsonl,
        summary_json=summary_json,
    )

    print("\nDone.")
    print(f"Total samples      : {total}")
    print(f"Refusals           : {refused}")
    print(f"Refusal rate       : {rate:.6f}")
    print(f"Avg infer time     : {avg_inference_time:.6f} sec")
    print(f"Run directory      : {run_dir}")
    print(f"Saved JSONL        : {out_jsonl}")
    print(f"Saved summary JSON : {summary_json}")
    print(f"Saved refusal CSV  : {refusal_rates_csv}")


if __name__ == "__main__":
    main()