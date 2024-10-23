# ollama-ocra-mini
##For using ocra mini 7b the steps are:     

1.** import libraries**
   import torch
   from transformers import LlamaForCausalLM, LlamaTokenizer    
   
2.** use model we want to use**
   model_name="pankajmathur/orca_mini_7b"
   tokenizer=LlamaTokenizer.from_pretrained(model_name)
   model=LlamaForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")    
   
3.**For text generation create a function **       

     
def generate_text(system,instruction,input=None):
  if input:
    prompt=f"###System:\n{system}\n\n###User:\n{instruction}\n\nInput:\n{input}\n\n###Response:\n"
  else:
    prompt=f"###System:\n{system}\n\n###User:\n{instruction}\n\n###Response:\n"
  tokens=tokenizer.encode(prompt)
  tokens=torch.tensor(tokens).unsqueeze(0)
  tokens=tokens.to("cuda")
  instance={'input_ids':tokens,'top_p':1.0,'temperature':0.7,'generate_len':1024,'top_k':50}
  length=len(tokens[0])
  with torch.no_grad():
    test=model.generate(
        input_ids=tokens,
        max_length=length+instance['generate_len'],   
        use_cache=True,
        do_sample=True,
        top_p=instance['top_p'],
        temperature=instance['temperature'],
        top_k=instance['top_k']   
    )  
    output= test[0][length:]   
    string=tokenizer.decode(output,skip_special_tokens=True)   
    return f'[!] Response: {string}'       

  4.Giving query      
  
  system='You are an AI assistant that follows instruction extremely well. Help me as much as you can.'   
  instruction='What is the future of AI'   
  print(generate_text(system,instruction))




  

