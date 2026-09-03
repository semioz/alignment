import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device == "cpu" else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length.")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define a pad token.")

    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    token_ids = [prompt + output for prompt, output in zip(prompt_ids, output_ids, strict=True)]
    max_length = max(map(len, token_ids))
    input_ids = torch.full(
        (len(token_ids), max_length), pad_token_id, dtype=torch.long
    )
    response_mask = torch.zeros((len(token_ids), max_length - 1), dtype=torch.bool)

    for index, (prompt, tokens) in enumerate(zip(prompt_ids, token_ids, strict=True)):
        input_ids[index, : len(tokens)] = torch.tensor(tokens)
        response_mask[index, len(prompt) - 1 : len(tokens) - 1] = True

    return {
        "input_ids": input_ids[:, :-1],
        "labels": input_ids[:, 1:],
        "response_mask": response_mask,
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    log_probs = torch.log_softmax(model(input_ids=input_ids).logits, dim=-1)
    output = {
        "log_probs": log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    }
    if return_token_entropy:
        output["token_entropy"] = -(log_probs.exp() * log_probs).sum(dim=-1)
    return output
