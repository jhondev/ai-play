from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-v0.1"

# - Create a new instance of the Tokenizer in a variable to process 
#   the text inputs and also to decode generated outputs later on
# - We're going to use the AutoTokenizer class to automatically 
#   download the tokenizer suitable for the model we're using
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")


# We're going to use the QuantizationConfig class to configure the quantization of the model
# Config: https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
  load_in_4bit=True, 
  bnb_4bit_use_double_quant=True, 
  bnb_4bit_quant_type="nf4", 
  bnb_4bit_compute_dtype=torch.bfloat16
)


# We're going to use the mistralai/Mistral-7B-v0.1 model, 
# that will be automatically downloaded from the Hugging Face's model hub
# We're going to use 4-bit dynamic quantization (https://huggingface.co/docs/transformers/main/en/main_classes/quantization) 
# to make the model run faster and use less memory, while sacrificing a little bit of accuracy
# You can run with the device_map="auto" option to use more of your GPU(s) 
# if you have one or many available on your device
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  quantization_config=quantization_config, 
  device_map="auto", 
  low_cpu_mem_usage=True
)


# Use the Tokenizer to encode the input text and store the result in a variable
input_text = "What is the best recipe for Pepperoni pizza?"
model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

# Generate text using the model and the tokenized input text
generated_text = model.generate(**model_inputs, max_length=20, pad_token_id=tokenizer.eos_token_id)

# Decode the generated text and print the result
result = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
print(result)