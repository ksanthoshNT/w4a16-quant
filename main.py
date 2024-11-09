import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.quanto import QuantizedModelForCausalLM, qint4

def quantize_model():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_id = "defog/llama-3-sqlcoder-8b"
    print(f"Loading model {model_id}...")
    
    # Load in BF16 to match original model's precision
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Starting quantization...")
    # Quantize the model to 4-bit weights
    # We exclude lm_head as it's typically better to keep it in full precision
    quantized_model = QuantizedModelForCausalLM.quantize(
        model,
        weights=qint4,  # 4-bit quantization for weights
        exclude=['lm_head']  # Exclude the language model head from quantization
    )
    
    print("Quantization complete!")

    # Save the quantized model
    output_dir = "./llama-3-sqlcoder-8b-w4a16"
    print(f"Saving quantized model to {output_dir}...")
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return quantized_model, tokenizer

def test_model(model, tokenizer):
    # Test the model with a simple SQL query generation
    prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    
    Generate a SQL query to answer this question: `Show me all users who signed up in 2023`
    
    DDL statements:
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        signup_date DATE
    );<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    The following SQL query best answers the question:
    ```sql"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.0,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nTest generation:")
    print(response)

if __name__ == "__main__":
    try:
        quantized_model, tokenizer = quantize_model()
        test_model(quantized_model, tokenizer)
        print("\nQuantization and testing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")