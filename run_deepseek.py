import sys
import torch
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_prompt():
    # Use sys.argv[1] if available, otherwise fall back to environment variable or default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = os.environ.get('PROMPT', os.environ.get('DEFAULT_PROMPT', "Tell me about DeepSeek."))
    
    prompt = prompt.strip('"')
    logging.info(f"Using prompt: {prompt}")
    return prompt

def main():
    try:
        logging.info("Starting DeepSeek inference script")
        
        # Get the prompt
        prompt = get_prompt()
        
        # Load the model from cached directory
        logging.info("Loading DeepSeek model from cache")
        model_path = "/workspace/model_cache"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        logging.info("Loading tokenizer from cache")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Generate response
        logging.info("Generating response")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Prepare output
        output = {
            "prompt": prompt,
            "response": response
        }

        # Save the output
        output_dir = Path('/outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'result.json'
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logging.info(f"Response generated and saved to {output_path}")
        logging.info(f"Response: {response}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        error_path = Path('/outputs/error.json')
        with open(error_path, 'w') as f:
            json.dump({"error": str(e)}, f)

if __name__ == "__main__":
    main()