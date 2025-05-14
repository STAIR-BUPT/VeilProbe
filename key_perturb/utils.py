from itertools import zip_longest
import json
import time
import os.path as osp
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer, LlamaForCausalLM, GPTNeoXForCausalLM, GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
# from peft import (
#     PeftModel,
#     PeftModelForCausalLM
# )

def read_txt_to_list_of_dict(fname: str):
    # fname should be a .jsonl
    results = []
    for l in open(fname).readlines():
        d = json.loads(l)
        results.append(d)
    return results


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
    ('a','b','c'), ('d','e','f'), ('g','x','x')"""
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def file_exists(wfname, overwrite):
    if osp.exists(wfname):
        if overwrite:
            print(f"Warning: {wfname} exists and will overwrite.")
            print("Staring in 5 seconds.")
            time.sleep(5)
            return False
        else:
            print(f"Error: {wfname} exists!")
            return True
    else:
        return False


def write_list_of_dict_to_jsonl(fwname: str, l_dict, overwrite=False):
    if file_exists(fwname, overwrite=overwrite):
        return

    fw = open(fwname, "w")
    for d in l_dict:
        fw.write(json.dumps(d, ensure_ascii=False) + "\n")
    fw.close()

    return


def genereate_wrapper(input_ids, model, generation_config, a_position_to_mask=None):
    attention_mask = torch.ones_like(input_ids)
    if a_position_to_mask is not None:
        # assert a_position_to_mask < input_ids.shape[-1]

        # set to 0: <unk>
        input_ids[0, a_position_to_mask] = 0
        attention_mask[0, a_position_to_mask] = 0
    # print('generate_wrapper:', input_ids)
    # print('generate_wrapper:', attention_mask)
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config
    )

    # generate_ids[0] is the start token <s>, automatically added by hf.
    # generate_ids[input_token_num] is the first token that is generated.

    # check text.
    # tokenizer.batch_decode(generated_ids)
    return generated_ids


def get_template(model_name):
    # vicuna
    # system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    # input_text = f"{system} USER:{query} ASSISTANT:"
    vicuna_template = {
        'prefix': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. User:",
        'postfix': " ASSISTANT:"
    }
    # llama2-chat
    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L358C19-L358C70
    llama_template = {
        'prefix': "[INST] ",
        'postfix': " [/INST]"
    }
    
    competion_templete = {
        'prefix': "",
        'postfix': ""
    }

    template = {
        'llama2-7b-chat': llama_template,
        'vicuna1.5-7b': vicuna_template,
        'pythia-6.9b': competion_templete,
        'llama-13b': competion_templete,
        'llama-7b': competion_templete,
        'pythia-2.8b': competion_templete,
        'pythia-12b':competion_templete,
        'pythia-1.4b':competion_templete,
        'pythia-1b':competion_templete,
        'gpt2': competion_templete,
        'gpt2-medium': competion_templete,
        'gpt2-large': competion_templete,
        'llama3.2-1b': competion_templete,
        # TO ADD.
    }[model_name]

    return template


def load_model(model_name):
    if model_name == 'llama-7b':
        model_path = '/root/autodl-tmp/llama-7b/'
        device_map = "auto"  # model parallel
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 4096
    
    elif model_name == 'llama-13b':
        model_path = '/root/autodl-tmp/llama-13b/'
        device_map = "auto"  # model parallel
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 5120
        
    elif model_name == 'gpt2':
        model_path = '/root/autodl-tmp/gpt2'
        device_map = "auto"  # model parallel
        model = GPT2LMHeadModel.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        block_name = "self.model.transformer.h"
        embedding_name = "self.model.transformer.wte"
        embedding_token_name = "self.model.transformer.wte.weight"
        vocab_size = model.config.vocab_size
        embed_dim = 768
        
    elif model_name == 'gpt2-medium':
        model_path = '/root/autodl-tmp/gpt2-medium'
        device_map = "auto"  # model parallel
        model = GPT2LMHeadModel.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        block_name = "self.model.transformer.h"
        embedding_name = "self.model.transformer.wte"
        embedding_token_name = "self.model.transformer.wte.weight"
        vocab_size = model.config.vocab_size
        embed_dim = 1024
    
    
    elif model_name == 'gpt2-large':
        model_path = '/root/autodl-tmp/gpt2-large'
        device_map = "auto"  # model parallel
        model = GPT2LMHeadModel.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        block_name = "self.model.transformer.h"
        embedding_name = "self.model.transformer.wte"
        embedding_token_name = "self.model.transformer.wte.weight"
        vocab_size = model.config.vocab_size
        embed_dim = 1280
        
        
    elif model_name == 'pythia-6.9b':
        model_path = "/path/to/file/pythia-6.9b"
        device = torch.device("cuda:0")
        model = GPTNeoXForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map = "auto",
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        embed_dim = 4096
        block_name = "self.model.gpt_neox.layers"
        embedding_name = "self.model.gpt_neox.embedding"
        embedding_token_name = "self.model.gpt_neox.embed_in.weight"
        vocab_size = model.config.vocab_size
    
    elif model_name == 'pythia-12b':
        model_path = "/root/autodl-tmp/pythia-12b/"
        device = torch.device("cuda:0")
        model = GPTNeoXForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map = "auto",
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        embed_dim = 5120
        block_name = "self.model.gpt_neox.layers"
        embedding_name = "self.model.gpt_neox.embedding"
        embedding_token_name = "self.model.gpt_neox.embed_in.weight"
        vocab_size = model.config.vocab_size
        
    
    elif model_name == 'pythia-2.8b':
        model_path = "/path/to/file/pythia-2.8b"
        device = torch.device("cuda:0")
        model = GPTNeoXForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map = "auto",
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        embed_dim = 2560
        block_name = "self.model.gpt_neox.layers"
        embedding_name = "self.model.gpt_neox.embedding"
        embedding_token_name = "self.model.gpt_neox.embed_in.weight"
        vocab_size = model.config.vocab_size
    
    
    elif model_name == 'pythia-1.4b':
        model_path = "/path/to/file/pythia-1.4b"
        device = torch.device("cuda:0")
        model = GPTNeoXForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map = "auto",
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        embed_dim = 2048
        block_name = "self.model.gpt_neox.layers"
        embedding_name = "self.model.gpt_neox.embedding"
        embedding_token_name = "self.model.gpt_neox.embed_in.weight"
        vocab_size = model.config.vocab_size
    
    elif model_name == 'pythia-1b':
        model_path = "/path/to/file/pythia-1b"
        device = torch.device("cuda:0")
        model = GPTNeoXForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map = "auto",
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        embed_dim = 2048
        block_name = "self.model.gpt_neox.layers"
        embedding_name = "self.model.gpt_neox.embedding"
        embedding_token_name = "self.model.gpt_neox.embed_in.weight"
        vocab_size = model.config.vocab_size
        
        
        
    elif model_name == 'llama2-7b-chat':
        model_path = "/mnt/sdb1/share/LLM_Models/Meta/Llama2/Llama-2-7b-chat-hf"
        # device_map = "auto"  # model parallel
        device = torch.device("cuda:0")
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            # device_map=device_map,
            low_cpu_mem_usage=True,
        ).to(device)

        state_dict = model.state_dict()
        for key in state_dict.keys():
            print(key)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 4096



    elif model_name == 'llama-13b':
        model_path = '/LLM_Models/Meta/Llama/llama-13b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 5120
    
    
    elif model_name == 'llama3.2-1b':
        model_path = '/root/autodl-tmp/Llama-3.2-1B/'
        device_map = "auto"  # model parallel
        config = AutoConfig.from_pretrained(model_name, rope_scaling={'type': 'linear', 'factor': 32.0})
        model =  AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
            config=config
        )

        print(model.hf_device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 2048
    return model, tokenizer, block_name, embedding_name, embedding_token_name, vocab_size, embed_dim