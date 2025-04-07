"""Functions that convert the model weights from the formats of other models
to that of SwiftTransformer.

This file follows the workflow described in 
`../../SwiftTransformer/scripts/converter_lib.py`. Please refer to that file 
for detailed explanation.

This file is divided into four parts:
 - Preprocessors: These functions preprocess the original state_dict and 
     convert them into the conventional OPT's format on 
     <https://github.com/facebookresearch/metaseq/tree/main/projects/OPT>
 - NameTranslators: They are used to convert the names of the weights from 
     different models (e.g. LLaMA2) to the format used by SwiftTransformer.
     The name translator is a function that takes a name and returns a new 
     name (or None if this weight should be ignored). The new name is the name 
     of the weight in SwiftTransformer (following OPT's naming convention)
 - Saver: The last part of converting weights. It takes a tensor_dict and 
     a name translator, translate the names of the weights, divide the weights 
     and save them to files.
 - Interface: Contains a function `convert_weights` which is the only 
     function provided to `downloader.py` in this file.
"""
import os
import re
import torch
import tqdm
import argparse

from glob import glob
from torch import nn
from typing import Callable, Dict, Optional, Tuple
from distserve.logger import init_logger

logger = init_logger(__name__)

#######################
#    Preprocessors    #
#######################

def preprocess_opt(tensor_dict: Dict[str, torch.Tensor])\
    -> Tuple[Dict[str, torch.Tensor], int, int]:

    num_q_heads = 0
    head_dim = 0
    if tensor_dict.get("decoder.embed_tokens.weight", None) is None:
        PREFIX = "model."
        regex = re.compile(r"model.decoder.layers.(\d+).fc1.weight")
    else:
        PREFIX = ""
        regex = re.compile(r"decoder.layers.(\d+).fc1.weight")
    FC1_BIAS = PREFIX + "decoder.layers.0.fc1.bias"
    LAYER_NORM = PREFIX + "decoder.final_layer_norm.weight"
    EMBED_TOKENS = PREFIX + "decoder.embed_tokens.weight"
    EMBED_POSITIONS = PREFIX + "decoder.embed_positions.weight"
    OUT_PROJ = PREFIX + "decoder.layers.{0}.self_attn.out_proj.weight"
    Q_WEIGHT = PREFIX + "decoder.layers.{0}.self_attn.q_proj.weight"
    K_WEIGHT = PREFIX + "decoder.layers.{0}.self_attn.k_proj.weight"
    V_WEIGHT = PREFIX + "decoder.layers.{0}.self_attn.v_proj.weight"
    Q_BIAS = PREFIX + "decoder.layers.{0}.self_attn.q_proj.bias"
    K_BIAS = PREFIX + "decoder.layers.{0}.self_attn.k_proj.bias"
    V_BIAS = PREFIX + "decoder.layers.{0}.self_attn.v_proj.bias"
    
    num_layers = max(int(regex.findall(x)[0]) for x in filter(regex.match, tensor_dict)) + 1

    ffn_inter_dim = tensor_dict[FC1_BIAS].size(0)
    head_dim = \
        64 if ffn_inter_dim <= 8192 else \
        80 if ffn_inter_dim == 10240 else \
        128
    num_q_heads = tensor_dict[LAYER_NORM].size(0) // head_dim

    tensor_dict[EMBED_POSITIONS] = tensor_dict[EMBED_POSITIONS][2:]
    tensor_dict["decoder.output_projection.weight"] = tensor_dict[EMBED_TOKENS]

    # Concatenate q_proj, k_proj, and v_proj
    # Transpose out_proj.weight and qkv_proj.weight
    for i in range(num_layers):
        tensor_dict[OUT_PROJ.format(i)] = \
            tensor_dict[OUT_PROJ.format(i)].T.contiguous()
        
        tensor_dict[f"decoder.layers.{i}.self_attn.qkv_proj.weight"] = \
            torch.cat([
                tensor_dict[Q_WEIGHT.format(i)].T,
                tensor_dict[K_WEIGHT.format(i)].T,
                tensor_dict[V_WEIGHT.format(i)].T,
            ], dim=1)
        del tensor_dict[Q_WEIGHT.format(i)]
        del tensor_dict[K_WEIGHT.format(i)]
        del tensor_dict[V_WEIGHT.format(i)]

        tensor_dict[f"decoder.layers.{i}.self_attn.qkv_proj.bias"] = \
            torch.cat([
                tensor_dict[Q_BIAS.format(i)],
                tensor_dict[K_BIAS.format(i)],
                tensor_dict[V_BIAS.format(i)]
            ])
        del tensor_dict[Q_BIAS.format(i)]
        del tensor_dict[K_BIAS.format(i)]
        del tensor_dict[V_BIAS.format(i)]

    assert num_q_heads > 0, "num_q_heads must be greater than 0"
    return tensor_dict, num_q_heads, head_dim

def preprocess_llama2(tensor_dict: Dict[str, torch.Tensor])\
    -> Tuple[Dict[str, torch.Tensor], int, int]:

    num_q_heads = 0
    head_dim = 0
    if tensor_dict.get("embed_tokens.weight", None) is None:
        PREFIX = "model."
        regex = re.compile(r"model.layers.(\d+).self_attn.q_proj.weight")
    else:
        PREFIX = ""
        regex = re.compile(r"layers.(\d+).self_attn.q_proj.weight")
    Q_WEIGHT = PREFIX + "layers.{0}.self_attn.q_proj.weight"
    K_WEIGHT = PREFIX + "layers.{0}.self_attn.k_proj.weight"
    V_WEIGHT = PREFIX + "layers.{0}.self_attn.v_proj.weight"
    O_WEIGHT = PREFIX + "layers.{0}.self_attn.o_proj.weight"

    end_layers = max(int(regex.findall(x)[0]) for x in filter(regex.match, tensor_dict)) + 1
    beg_layers = min(int(regex.findall(x)[0]) for x in filter(regex.match, tensor_dict))
    
    head_dim = 128
    num_q_heads = tensor_dict[Q_WEIGHT.format(24)].size(0) // head_dim

    # Coallesce wq, qk, qv into one tensor, layers.{i}.attention.wqkv.weight

    for i in range(beg_layers,end_layers):
        q = tensor_dict[Q_WEIGHT.format(i)].T  # [hidden_size, num_q_heads*head_dim]
        k = tensor_dict[K_WEIGHT.format(i)].T  # [hidden_size, num_kv_heads*head_dim]
        v = tensor_dict[V_WEIGHT.format(i)].T  # [hidden_size, num_kv_heads*head_dim]
        wqkv = torch.cat([q, k, v], dim=1)    # [hidden_size, (num_q_heads+2*num_kv_heads)*head_dim]
        tensor_dict[f"layers.{i}.attention.wqkv.weight"] = wqkv
        del tensor_dict[Q_WEIGHT.format(i)]
        del tensor_dict[K_WEIGHT.format(i)]
        del tensor_dict[V_WEIGHT.format(i)]

    # Transpose wo
    for i in range(beg_layers,end_layers):
        tensor_dict[O_WEIGHT.format(i)] = \
            tensor_dict[O_WEIGHT.format(i)].T.contiguous()  # [num_q_heads*head_dim, hidden_size]

    assert num_q_heads > 0, "num_q_heads must be greater than 0"
    return tensor_dict, num_q_heads, head_dim

def preprocess_gpt2(tensor_dict: Dict[str, torch.Tensor])\
    -> Tuple[Dict[str, torch.Tensor], int, int]:
    num_q_heads = 0
    head_dim = 0
    num_layers = 1 + max(map(lambda x: int(x[0]), filter(lambda x: len(x) > 0, (re.compile(r"h.(\d+).attn.c_attn.weight").findall(x) for x in tensor_dict.keys()))))
    ffn_inter_dim = tensor_dict["h.0.mlp.c_fc.bias"].size(0)
    head_dim = \
        64 if ffn_inter_dim <= 8192 else \
        80 if ffn_inter_dim == 10240 else \
        128
    num_q_heads = tensor_dict["ln_f.weight"].size(0) // head_dim
    tensor_dict["decoder.output_projection.weight"] = tensor_dict["wte.weight"]

    for key, tensor in tensor_dict.items():
        if re.compile(r"h.(\d+).mlp.(c_fc|c_proj).weight").match(key):
            tensor_dict[key] = tensor.T

    assert num_q_heads > 0, "num_q_heads must be greater than 0"
    return tensor_dict, num_q_heads, head_dim

PREPROCESSOR = {
    "opt": preprocess_opt,
    "llama": preprocess_llama2,
    "gpt2": preprocess_gpt2
}

#########################
#    NameTranslators    #
#########################

"""
decoder.embed_tokens.weight                   decoder.embed_tokens.weight                  torch.Size([50272, 768])
decoder.embed_positions.weight                decoder.embed_positions.weight               torch.Size([2048, 768])
decoder.final_layer_norm.weight               decoder.layer_norm.weight                    torch.Size([768])
decoder.final_layer_norm.bias                 decoder.layer_norm.bias                      torch.Size([768])
decoder.output_projection.weight              decoder.output_projection.weight             torch.Size([50272, 768])
decoder.layers.0.self_attn.out_proj.weight    decoder.layers.0.self_attn.out_proj.weight   torch.Size([768, 768])
decoder.layers.0.self_attn.out_proj.bias      decoder.layers.0.self_attn.out_proj.bias     torch.Size([768])
decoder.layers.0.self_attn_layer_norm.weight  decoder.layers.0.self_attn_layer_norm.weight torch.Size([768])
decoder.layers.0.self_attn_layer_norm.bias    decoder.layers.0.self_attn_layer_norm.bias   torch.Size([768])
decoder.layers.0.fc1.weight                   decoder.layers.0.fc1.weight                  torch.Size([3072, 768])
decoder.layers.0.fc1.bias                     decoder.layers.0.fc1.bias                    torch.Size([3072])
decoder.layers.0.fc2.weight                   decoder.layers.0.fc2.weight                  torch.Size([768, 3072])
decoder.layers.0.fc2.bias                     decoder.layers.0.fc2.bias                    torch.Size([768])
decoder.layers.0.final_layer_norm.weight      decoder.layers.0.final_layer_norm.weight     torch.Size([768])
decoder.layers.0.final_layer_norm.bias        decoder.layers.0.final_layer_norm.bias       torch.Size([768])
decoder.layers.0.self_attn.qkv_proj.weight    decoder.layers.0.self_attn.qkv_proj.weight   torch.Size([768, 2304])
decoder.layers.0.self_attn.qkv_proj.bias      decoder.layers.0.self_attn.qkv_proj.bias     torch.Size([2304])
"""
def optNameTranslator(name: str) -> Optional[str]:
    if name == "lm_head.weight":
        return None
    if re.match(r"model", name):
        name = name[6:]
    name_mapping_table = [
        (re.compile(r"decoder.final_layer_norm.weight"), "decoder.layer_norm.weight"),
        (re.compile(r"decoder.final_layer_norm.bias"), "decoder.layer_norm.bias"),
    ]
    for (regex, newname) in name_mapping_table:
        match = regex.match(name)
        if match:
            return newname
    return name

def llama2NameTranslator(name: str) -> Optional[str]:
    ignore_regex = re.compile(r"self_attn.rotary_emb.inv_freq")
    if ignore_regex.search(name):
        return None
    if re.match(r"model", name):
        name = name[6:]
    name_mapping_table = [
        (re.compile(r"layers.(?P<layer>\d+).self_attn.o_proj.weight"), "decoder.layers.{layer}.self_attn.out_proj.weight"),
        (re.compile(r"layers.(?P<layer>\d+).mlp.gate_proj.weight"), "decoder.layers.{layer}.fc1.weight"),
        (re.compile(r"layers.(?P<layer>\d+).mlp.down_proj.weight"), "decoder.layers.{layer}.fc2.weight"),
        (re.compile(r"layers.(?P<layer>\d+).mlp.up_proj.weight"), "decoder.layers.{layer}.fc3.weight"),
        (re.compile(r"layers.(?P<layer>\d+).input_layernorm.weight"), "decoder.layers.{layer}.self_attn_layer_norm.weight"),
        (re.compile(r"layers.(?P<layer>\d+).post_attention_layernorm.weight"), "decoder.layers.{layer}.final_layer_norm.weight"),
        (re.compile(r"layers.(?P<layer>\d+).attention.wqkv.weight"), "decoder.layers.{layer}.self_attn.qkv_proj.weight"),
        (re.compile(r"embed_tokens.weight"), "decoder.embed_tokens.weight"),
        (re.compile(r"lm_head.weight"), "decoder.output_projection.weight"),
        (re.compile(r"norm.weight"), "decoder.layer_norm.weight")
    ]
    for (regex, newname) in name_mapping_table:
        match = regex.match(name)
        if match:
            return newname.format(**match.groupdict())
    assert False, f"Cannot find a match for {name} when translating name"

"""
wte.weight                                    decoder.embed_tokens.weight                  torch.Size([50272, 768])
wpe.weight                                    decoder.embed_positions.weight               torch.Size([2048, 768])
ln_f.weight                                   decoder.layer_norm.weight                    torch.Size([768])
ln_f.bias                                     decoder.layer_norm.bias                      torch.Size([768])
decoder.output_projection.weight              decoder.output_projection.weight             torch.Size([50272, 768])
h.0.attn.c_proj.weight                        decoder.layers.0.self_attn.out_proj.weight   torch.Size([768, 768])
h.0.attn.c_proj.bias                          decoder.layers.0.self_attn.out_proj.bias     torch.Size([768])
h.0.ln_1.weight                               decoder.layers.0.self_attn_layer_norm.weight torch.Size([768])
h.0.ln_1.bias                                 decoder.layers.0.self_attn_layer_norm.bias   torch.Size([768])
h.0.mlp.c_fc.weight                           decoder.layers.0.fc1.weight                  torch.Size([3072, 768])
h.0.mlp.c_fc.bias                             decoder.layers.0.fc1.bias                    torch.Size([3072])
h.0.mlp.c_proj.weight                         decoder.layers.0.fc2.weight                  torch.Size([768, 3072])
h.0.mlp.c_proj.bias                           decoder.layers.0.fc2.bias                    torch.Size([768])
h.0.ln_2.weight                               decoder.layers.0.final_layer_norm.weight     torch.Size([768])
h.0.ln_2.bias                                 decoder.layers.0.final_layer_norm.bias       torch.Size([768])
h.0.attn.c_attn.weight                        decoder.layers.0.self_attn.qkv_proj.weight   torch.Size([768, 2304])
h.0.attn.c_attn.bias                          decoder.layers.0.self_attn.qkv_proj.bias     torch.Size([2304])

h.0.attn.bias is the self attn mask and should (probably) be removed
"""
def gpt2NameTranslator(name: str) -> Optional[str]:
    ignore_list = [
        re.compile(r"h.(\d+).attn.bias"),
    ]
    for regex in ignore_list:
        if regex.match(name):
            return None
    name_mapping_table = [
        (re.compile(r"h.(?P<layer>\d+).attn.c_proj.weight"), "decoder.layers.{layer}.self_attn.out_proj.weight"),
        (re.compile(r"h.(?P<layer>\d+).attn.c_proj.bias"), "decoder.layers.{layer}.self_attn.out_proj.bias"),
        (re.compile(r"h.(?P<layer>\d+).ln_1.weight"), "decoder.layers.{layer}.self_attn_layer_norm.weight"),
        (re.compile(r"h.(?P<layer>\d+).ln_1.bias"), "decoder.layers.{layer}.self_attn_layer_norm.bias"),
        (re.compile(r"h.(?P<layer>\d+).mlp.c_fc.weight"), "decoder.layers.{layer}.fc1.weight"),
        (re.compile(r"h.(?P<layer>\d+).mlp.c_fc.bias"), "decoder.layers.{layer}.fc1.bias"),
        (re.compile(r"h.(?P<layer>\d+).mlp.c_proj.weight"), "decoder.layers.{layer}.fc2.weight"),
        (re.compile(r"h.(?P<layer>\d+).mlp.c_proj.bias"), "decoder.layers.{layer}.fc2.bias"),
        (re.compile(r"h.(?P<layer>\d+).ln_2.weight"), "decoder.layers.{layer}.final_layer_norm.weight"),
        (re.compile(r"h.(?P<layer>\d+).ln_2.bias"), "decoder.layers.{layer}.final_layer_norm.bias"),
        (re.compile(r"h.(?P<layer>\d+).attn.c_attn.weight"), "decoder.layers.{layer}.self_attn.qkv_proj.weight"),
        (re.compile(r"h.(?P<layer>\d+).attn.c_attn.bias"), "decoder.layers.{layer}.self_attn.qkv_proj.bias"),
        (re.compile(r"wte.weight"), "decoder.embed_tokens.weight"),
        (re.compile(r"wpe.weight"), "decoder.embed_positions.weight"),
        (re.compile(r"ln_f.weight"), "decoder.layer_norm.weight"),
        (re.compile(r"ln_f.bias"), "decoder.layer_norm.bias"),
        (re.compile(r"decoder.output_projection.weight"), "decoder.output_projection.weight")
    ]
    for (regex, newname) in name_mapping_table:
        match = regex.match(name)
        if match:
            return newname.format(**match.groupdict())
    assert False, f"Cannot find a match for {name} when translating name"

NAME_TRANSLATOR = {
    "opt": optNameTranslator,
    "llama": llama2NameTranslator,
    "gpt2": gpt2NameTranslator
}

###############
#    Saver    #
###############

def saveTensorToFile(filename: str, key: str, tensor: torch.Tensor):
    """Save a tensor to a file.
    The file can be loaded by torch.jit.load(). The file contains a single
    tensor whose key is "key"
    
    https://discuss.pytorch.org/t/load-tensors-saved-in-python-from-c-and-vice-versa/39435/8
    """
    class TensorContainer(nn.Module):
        def __init__(self, key: str, tensor: torch.Tensor):
            super().__init__()
            setattr(self, key, tensor)

    container = TensorContainer(key, tensor.clone()) # clone() is needed or the whole (undivided) tensor will be saved
    torch.jit.script(container).save(filename)

def divideWeightAndSave(
    output_dir: str, 
    tensor_dict: Dict[str, torch.Tensor], 
    name_translator: Callable[[str], Optional[str]], 
    num_q_heads: int, 
    head_dim: int
):
    """The last step in convertWeight(). It takes a tensor_dict and a name 
    translator, translate the names of the weights, divide the weights and 
    save them to files"""
    
    def divideTensorAndStore(new_key, value, dim: int):
        """divide a tensor along a dimension into 8 pieces and save the divided 
        tensors to files"""
        assert value.size(dim) % 8 == 0, f"Cannot divide {new_key} along dim={dim} because the size of the dimension is not divisible by 8"
        value = torch.split(value, value.size(dim) // 8, dim=dim)
        # save the tensor to file
        for i in range(len(value)):
            filename = os.path.join(output_dir, f"{new_key}.tp_{i}.pt")
            saveTensorToFile(filename, new_key, value[i])

    # storeQKVKernelOrBias: divide the QKV kernel or bias and save them to files
    def storeQKVKernelOrBias(key: str, qkvs: torch.Tensor, split_dim: int):
        qkvs = qkvs.view(qkvs.size(0), -1, head_dim) if split_dim == 1 else qkvs.view(-1, head_dim)
        num_kv_heads = (qkvs.size(split_dim)-num_q_heads) // 2
        # qkvs: [hidden_size, (num_q_heads+2*num_kv_heads), head_dim] (when converting QKV kernel)
        # or, [(num_q_heads+2*num_kv_heads), head_dim] (when converting QKV bias)

        # Deal with cases where num_q_heads or num_kv_heads is not divisible by 8, like in OPT-125M
        q_heads_in_each_tensor = num_q_heads // 8 if num_q_heads%8 == 0 else [ num_q_heads // 8 ] * 7 + [ num_q_heads - (num_q_heads // 8) * 7 ]
        kv_heads_in_each_tensor = num_kv_heads // 8 if num_kv_heads%8 == 0 else [ num_kv_heads // 8 ] * 7 + [ num_kv_heads - (num_kv_heads // 8) * 7 ]

        # split qkvs into 3 parts: q, k, v
        qs = qkvs.narrow(split_dim, 0, num_q_heads).split(q_heads_in_each_tensor, split_dim)
        ks = qkvs.narrow(split_dim, num_q_heads, num_kv_heads).split(kv_heads_in_each_tensor, split_dim)
        vs = qkvs.narrow(split_dim, num_q_heads+num_kv_heads, num_kv_heads).split(kv_heads_in_each_tensor, split_dim)

        # save the tensors to files
        for i in range(8):
            filename = os.path.join(output_dir, f"{key}.tp_{i}.pt")
            saveTensorToFile(filename, key, torch.cat([qs[i], ks[i], vs[i]], dim=split_dim))

    # The following regexs define tensors in the standarized tensor dict
    # Note that not all tensors listed below need to present. For example,
    # LLaMA2 does not have decoder.embed_positions.weight since it used RoPE.

    # Tensors that need to be divided along dim=0
    to_divide_by_dim0_regex = re.compile("|".join([
        "decoder.layers.(\d+).fc1.weight",      # [ffn_inter_dim, hidden_size]
        "decoder.layers.(\d+).fc1.bias",        # [ffn_inter_dim]
        "decoder.layers.(\d+).fc3.weight",      # [ffn_inter_dim, hidden_size]
        "decoder.layers.(\d+).self_attn.out_proj.weight",   # [(num_q_heads*head_dim), hidden_size]
        "decoder.layers.(\d+).self_attn.qkv_proj.bias"      # [(num_q_heads+2*num_kv_heads)*head_dim]
                                         ]))
    # Tensors that need to be divided along dim=1
    to_divide_by_dim1_regex = re.compile("|".join([
        "decoder.layers.(\d+).fc2.weight"       # [hidden_size, ffn_inter_dim]
                                         ]))
    # Tensors that need to be replicated among all tensor parallel workers
    to_replicate_regex = re.compile("|".join([
        "decoder.embed_tokens.weight",          # [vocab_size, hidden_size]
        "decoder.embed_positions.weight",       # [max_positions, hidden_size]
        "decoder.output_projection.weight",     # [vocab_size, hidden_size]
        "decoder.layer_norm.(weight|bias)",     # [hidden_size] (The final layernorm)
        "decoder.layers.(\d+).self_attn_layer_norm.(weight|bias)",  # [hidden_size]
        "decoder.layers.(\d+).final_layer_norm.(weight|bias)",      # [hidden_size]
        "decoder.layers.(\d+).self_attn.out_proj.bias",             # [hidden_size]
        "decoder.layers.(\d+).fc2.bias"                    # [hidden_size]
                                    ]))
    
    # And we have two special tensors:
    #   - decoder.layers.{layer_id}.self_attn.qkv_proj.weight   [hidden_size, (num_q_heads+2*num_kv_heads)*head_dim]
    #   - decoder.layers.{layer_id}.self_attn.qkv_proj.bias     [(num_q_heads+2*num_kv_heads)*head_dim]
    # which need to be handled separately

    summary = ""
    for key, tensor in tqdm.tqdm(tensor_dict.items()):
        new_key = name_translator(key)
        if new_key == None:
            continue
        summary += f"Loaded {key} as {new_key} with shape {tensor.shape}\n"
        if "self_attn.qkv_proj.weight" in new_key:
            storeQKVKernelOrBias(new_key, tensor, split_dim=1)
        elif "self_attn.qkv_proj.bias" in new_key:
            storeQKVKernelOrBias(new_key, tensor, split_dim=0)
        elif to_divide_by_dim0_regex.search(new_key):
            divideTensorAndStore(new_key, tensor, dim=0)
        elif to_divide_by_dim1_regex.search(new_key):
            divideTensorAndStore(new_key, tensor, dim=1)
        elif to_replicate_regex.search(new_key):
            filename = os.path.join(output_dir, f"{new_key}.pt")
            saveTensorToFile(filename, new_key, tensor)
        else:
            assert False, f"Cannot find a match for {new_key} when dispatching tensors"
    print(summary)

###################
#    Interface    #
###################

def convert_weights(
    input: str, 
    output: str, 
    dtype: torch.dtype, 
    model: str
) -> None :
    """Function used by `downloader.py` to convert weights"""
    os.makedirs(output, exist_ok=True)
    print(f"Converting {input} into torch.jit.script format")

    # Load the state dict (tensor_dict)
    # If the whole model is saved in a single file, then load the state dict directly
    # otherwise, load them separately and merge them into a single state dict
    bin_files = glob(os.path.join(input, '*.bin'))
    #logger.info(f"Find bin files : {bin_files}.") 
    #input_files = glob(input)
    if len(bin_files) == 0:
        ValueError(f"Input {input} does not match any files")
        print(f"Input {input} does not match any files")
        exit(1)
    
    # Load file(s)
    state_dict = {}
    for file in bin_files:
        print(f"Loading {file}")
        state_dict.update(torch.load(file, torch.device("cpu")))

    # Change dtype
    for key in state_dict:
        state_dict[key] = state_dict[key].to(dtype)

    # Preprocess
    print("Preprocessing")
    preprocessor = PREPROCESSOR[model]
    print(state_dict)
    tensor_dict, num_q_heads, head_dim = preprocessor(state_dict)

    # The final step: divide the weights and save them to files
    print("Resharding and saving weights")
    name_translator = NAME_TRANSLATOR[model]
    divideWeightAndSave(output, tensor_dict, name_translator, num_q_heads, head_dim)
    
#######################
#    CLI Interface    #
#######################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert weights from other models to SwiftTransformer's format")
    parser.add_argument("--input", type=str, required=True, help="Path to input weights")
    parser.add_argument("--output", type=str, required=True, help="Path to output weights")
    parser.add_argument("--dtype", type=str, default="float16", help="dtype of the weights")
    parser.add_argument("--model", type=str, required=True, help="Model type")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32
    }[args.dtype]

    convert_weights(args.input, args.output, dtype, args.model)
