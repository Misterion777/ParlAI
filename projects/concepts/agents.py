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
from parlai.core.torch_agent import History
from parlai.utils.concepts import get_knowledge
import logging


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

hooks_created = False
class AttentionOutTransformerAgentMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_enc_layers = len(self.model.encoder.layers)
        self.num_dec_layers = len(self.model.decoder.layers)
    
        # Register hooks to record outputs
        encoder_module_map = {
            'layers': TransformerEncoder,
            'attentions': MultiHeadAttention,
        }
        decoder_module_map = {
            'layers': TransformerDecoder,
            'attentions': MultiHeadAttention,
        }
        global hooks_created
        if not hooks_created:
            self.hooks = {                        
                'encoder': self._register_series_of_hooks(
                    model=self.model.encoder, module_map=encoder_module_map
                ),
                'decoder': self._register_series_of_hooks(
                    model=self.model.decoder, module_map=decoder_module_map
                ),      
                'embeddings':OutputRecorder(self)
            }
            hooks_created = True
        else:            
            self.hooks = None
            print('Can be hooked only once!')  


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

    def extract_attentions(self):
        dec_outputs = self.hooks['decoder']['layers'].outputs
        enc_outputs = self.hooks['encoder']['layers'].outputs

        encoder_attn = []
        decoder_attn = []
        cross_attn = []

        i = 0
        print('total dec runs: ',len(dec_outputs))
        while i < len(dec_outputs):
            dec_attn = torch.cat([layer['self_attn'].unsqueeze(0) for layer in dec_outputs[i]])
            cr_attn = torch.cat([layer['encoder_attn'].unsqueeze(0) for layer in dec_outputs[i]])
            dec_tmp_attn = []
            cr_tmp_attn = []
            while dec_attn.size()[2] == 1: # skip layer, head
                print('dec_attn size: ',dec_attn.size())
                print('cr_attn size: ',cr_attn.size())
                dec_tmp_attn.append(dec_attn)
                cr_tmp_attn.append(cr_attn)
                i += 1
                if i >= len(dec_outputs):
                    break
                dec_attn = torch.cat([layer['self_attn'].unsqueeze(0) for layer in dec_outputs[i]])        
                cr_attn = torch.cat([layer['encoder_attn'].unsqueeze(0) for layer in dec_outputs[i]])        
                if dec_attn.size()[3] == 1:
                    i -= 1
                    break
            
            if len(dec_tmp_attn) == 0:
                decoder_attn.append(dec_attn)
                cross_attn.append(cr_attn)
            else:
                max_dim = dec_tmp_attn[-1].size(-1)
                assert len(dec_tmp_attn) == max_dim, f"{len(dec_tmp_attn)}!={max_dim}" # insure square mat
                dec_tmp_attn = cat_pad(dec_tmp_attn,max_dim)
                cr_tmp_attn = torch.cat(cr_tmp_attn,dim=2)
                
                decoder_attn.append(dec_tmp_attn)
                cross_attn.append(cr_tmp_attn)

            i +=1
        
        encoder_attn = [[layer['self_attn'].unsqueeze(0).cpu() for layer in run] for run in enc_outputs]  
        decoder_attn = [[layer.unsqueeze(0).cpu() for layer in run] for run in decoder_attn] 
        cross_attn = [[layer.unsqueeze(0).cpu() for layer in run] for run in cross_attn] 

        return encoder_attn,decoder_attn,cross_attn

    def extract_input_tokens(self):
        enc_inputs = self.hooks['encoder']['layers'].inputs        

        enc_in = []             
        print('encoder inputs len:',len(enc_inputs))
        for run_tokens in enc_inputs:
            print('Size of tokens: ',run_tokens.size())
            run_tokens = run_tokens.squeeze(0)
            enc_in.append([self.dict.ind2tok[ind.item()] for ind in run_tokens])                
        return enc_in
                

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


class ConceptsHistory(History):
    def get_history_vec(self,concepts=None):
        """
        Return a vectorized version of the history.
        """
        if len(self.history_vecs) == 0:
            return None

        # vec type is a list
        history = []
        for vec in self.history_vecs[:-1]:
            history += [vec]
            history += [self.delimiter_tok]
        history += [self.history_vecs[-1]]
        if self.temp_history is not None:
            history.extend([self.parse(self.temp_history)])
        if concepts is not None:
            # print(f"Concepts: {concepts}")
            history.extend([self.parse(concepts)])
            # print(f"Tokens: {history[-1]}")
        if self._global_end_token is not None:
            history += [[self._global_end_token]]

        history = sum(history, [])
        if self.reversed:
            history = list(reversed(history))

        return history

class ConceptsTransformerAgent(TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('ConceptsTransformer arguments')
        agent.add_argument(
            '--extract-from-history',
            type='bool',
            default=True,
            help="whether to extract concepts from whole history. If False will extract concepts only from latest message.",
        )        
        return agent

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overridden if a more complex history is required.
        """
        return ConceptsHistory

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs['full_text'] = history_string
            knowledge = None
            if self.opt.get("extract_runtime"):
                if self.opt.get("extract_from_history"):
                    extract_from = history_string.replace("your persona:","")
                else:
                    extract_from = obs["text"]

                knowledge = get_knowledge(extract_from,limit=2,compare_all_text=True)
                # print("Extracted concepts:")
                # print(knowledge)
                # obs.force_set("knowledge",knowledge)
                obs.force_set("text",obs["text"] + knowledge)

            if history_string:
                obs['text_vec'] = history.get_history_vec(knowledge)
                obs['full_text_vec'] = history.get_history_vec(knowledge)

        # check truncation
        if obs.get('text_vec') is not None:
            truncate_left = not self.history_reversed
            text_length = len(obs['text_vec'])
            truncated_vec = self._check_truncate(
                obs['text_vec'], truncate, truncate_left
            )
            obs.force_set('context_original_length', text_length)
            obs.force_set('context_truncate_rate', text_length != len(truncated_vec))
            obs.force_set(
                'context_truncated_length', max(text_length - len(truncated_vec), 0)
            )
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))

        return obs