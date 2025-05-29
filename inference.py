import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a fine-tuned GPT-2 model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="results",
        help="Path to the fine-tuned model folder under cc-finetuning"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="The initial text to prompt the model"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum total length of generated text"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="How many independent samples to generate"
    )
    args = parser.parse_args()

    # Load tokenizer and model from the specified directory
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.eval()

    # Encode prompt and generate
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode and print each result
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"=== GENERATED {i+1} ===")
        print(text)

if __name__ == "__main__":
    main()
