#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Code for distilling a transformer/generator model.
"""

from typing import Optional
from parlai.core.params import ParlaiParser
from typing import Any, Dict, List, Type, Union

import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from parlai.agents.transformer.modules import (
    MultiHeadAttention,
    TransformerDecoder,
    TransformerEncoder,
)
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.loader import register_agent


def cat_pad(tensor_list,max_dim):
    padded = [F.pad(tensor,(0,max_dim-tensor.size(-1))) for tensor in tensor_list]
    return torch.cat(padded,dim=2)             


class OutputRecorder:
    """
    Saves all outputs from modules that it is registered to.
    """

    def __init__(self,agent):
        self.agent = agent
        self.inputs = []
        self.outputs = []

    def __call__(self, module: nn.Module, module_in: Any, module_out: Any):
        tokens_id = 0
        if isinstance(module, TransformerEncoder):
            module_in = module_in[tokens_id]
            module_out = self.agent.extract_encoder_attention()
        elif isinstance(module, TransformerDecoder):
            module_in = module_in[tokens_id]
            module_out = self.agent.extract_decoder_attention()

        self.inputs.append(module_in)
        self.outputs.append(module_out)

    def clear(self):
        self.inputs = []
        self.outputs = []

class AttentionOutTransformerAgentMixin():
    def build_model(self):

        model = super().build_model()

        self.num_enc_layers = len(model.encoder.layers)
        self.num_dec_layers = len(model.decoder.layers)
    
        # Register hooks to record outputs
        encoder_module_map = {
            'layers': TransformerEncoder,
            'attentions': MultiHeadAttention,
        }
        decoder_module_map = {
            'layers': TransformerDecoder,
            'attentions': MultiHeadAttention,
        }
        self.hooks = {                        
            'encoder': self._register_series_of_hooks(
                model=model.encoder, module_map=encoder_module_map
            ),
            'decoder': self._register_series_of_hooks(
                model=model.decoder, module_map=decoder_module_map
            ),      
            'embeddings':OutputRecorder(self)
        }
        model.embeddings.register_forward_hook(
            self.hooks['embeddings']
        )

        return model

    def _register_series_of_hooks(
        self, model: nn.Module, module_map: Dict[str, Type[nn.Module]]
    ) -> Dict[str, OutputRecorder]:
        """
        Register hooks in modules of the model, given the mapping of module types.

        `module_map` is a dict whose keys are module-type names and whose values are
        module types. For each module type, during each forward pass of `model`, all
        outputs of modules of that type will be saved to `hooks[module_type].outputs`.
        """
        hooks = {}
        for module_name, module_type in module_map.items():
            hooks[module_name] = OutputRecorder(self)
            for module in model.modules():
                if isinstance(module, module_type):
                    module.register_forward_hook(hooks[module_name])
        return hooks

    def extract_embedding_outputs(self) -> Dict[str, torch.Tensor]:
        """
        Extract out the encoder and decoder embedding outputs.
        """
        assert len(self.hooks['embeddings'].outputs) == 2
        return {
            'encoder': self.hooks['embeddings'].outputs[0],
            'decoder': self.hooks['embeddings'].outputs[1],
        }

    def extract_attentions(self):
        dec_outputs = self.hooks['decoder']['layers'].outputs
        enc_outputs = self.hooks['encoder']['layers'].outputs

        encoder_attn = []
        decoder_attn = []
        cross_attn = []

        i = 0
        while i < len(dec_outputs):
            dec_attn = torch.cat([layer['self_attn'].unsqueeze(0) for layer in dec_outputs[i]])
            cr_attn = torch.cat([layer['encoder_attn'].unsqueeze(0) for layer in dec_outputs[i]])
            dec_tmp_attn = []
            cr_tmp_attn = []
            while dec_attn.size()[2] == 1: # skip layer, head
                dec_tmp_attn.append(dec_attn)
                cr_tmp_attn.append(cr_attn)
                i += 1
                if i >= len(dec_outputs):
                    break
                dec_attn = torch.cat([layer['self_attn'].unsqueeze(0) for layer in dec_outputs[i]])        
                cr_attn = torch.cat([layer['encoder_attn'].unsqueeze(0) for layer in dec_outputs[i]])        
            
            if len(dec_tmp_attn) == 0:
                decoder_attn.append(dec_attn)
                cross_attn.append(cr_attn)
            else:
                max_dim = dec_tmp_attn[-1].size(-1)
                assert len(dec_tmp_attn) == max_dim # insure square mat
                dec_tmp_attn = cat_pad(dec_tmp_attn,max_dim)
                cr_tmp_attn = torch.cat(cr_tmp_attn,dim=2)
                
                decoder_attn.append(dec_tmp_attn)
                cross_attn.append(cr_tmp_attn)

            i +=1
        
        encoder_attn = [[layer['self_attn'].unsqueeze(0) for layer in run] for run in enc_outputs]  
        decoder_attn = [[layer.unsqueeze(0) for layer in run] for run in decoder_attn] 
        cross_attn = [[layer.unsqueeze(0) for layer in run] for run in cross_attn] 

        return encoder_attn,decoder_attn,cross_attn

    def extract_tokens(self):
        enc_in = []
        dec_in = []        

        for run_tokens in self.hooks['encoder']['layers'].inputs:
            run_tokens = run_tokens.squeeze(0)
            enc_in.append([self.dict.ind2tok[ind.item()] for ind in run_tokens])
        for run_tokens in self.hooks['decoder']['layers'].inputs:
            run_tokens = run_tokens.squeeze(0)
            dec_in.append([self.dict.ind2tok[ind.item()] for ind in run_tokens])
                
        return enc_in,dec_in
                

    def extract_encoder_attention(self):
        assert len(self.hooks['encoder']['attentions'].outputs) == self.num_enc_layers
        output_idx = 2  # The position of the attention matrix among the outputs
        output = [
                {
                    'self_attn': self.hooks['encoder']['attentions'].outputs[layer_idx][
                        output_idx
                    ]
                }
                for layer_idx in range(self.num_enc_layers)
            ]
        self.hooks['encoder']['attentions'].clear()
        return output


    def extract_decoder_attention(self):
        assert len(self.hooks['decoder']['attentions'].outputs) == 2 * self.num_dec_layers
        output_idx = 2  # The position of the attention matrix among the outputs
        result = [
                {
                    'self_attn': self.hooks['decoder']['attentions'].outputs[2 * layer_idx][
                        output_idx
                    ],
                    'encoder_attn': self.hooks['decoder']['attentions'].outputs[
                        2 * layer_idx + 1
                    ][output_idx],
                }
                for layer_idx in range(self.num_dec_layers)
            ]
        self.hooks['decoder']['attentions'].clear()
        return result

    def clear_hooks(self):
        self._clear_hook_outputs(self.hooks)

    def _clear_hook_outputs(self, hooks: Union[Dict[str, Any], OutputRecorder]):
        """
        Recursively clear outputs from all hooks.
        """
        if isinstance(hooks, dict):
            for subhooks in hooks.values():
                self._clear_hook_outputs(subhooks)
        else:
            # `hooks` is an OutputRecorder
            hooks.clear()

@register_agent("return_attn_transformer_agent")
class AttentionOutTransformerAgent(AttentionOutTransformerAgentMixin, TransformerGeneratorAgent):
    pass