from transformers import GPTNeoXForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
# Note: This model takes 15 GB of Vram when loaded in 8bit
model = GPTNeoXForCausalLM.from_pretrained(
  "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",device_map="auto", load_in_8bit=True)
# for cpu ver
# model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", torch_dtype=torch.bfloat16)
message = "<|prompter|>What is a meme, and what's the history behind this word?<|endoftext|><|assistant|>"
inputs = tokenizer(message, return_tensors="pt").to(model.device)
tokens = model.generate(**inputs,  max_new_tokens=1000, do_sample=True, temperature=0.8)
tokenizer.decode(tokens[0])