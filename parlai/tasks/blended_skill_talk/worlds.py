#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import json
import random
import pickle
from pathlib import Path

from parlai.tasks.blended_skill_talk.agents import raw_data_path, safe_personas_path
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.utils.io import PathManager


def get_contexts_data(opt, shared=None):
    if shared and 'contexts_data' in shared:
        return shared['contexts_data']
    return _load_personas(opt=opt)


def _load_personas(opt):
    print('[ loading personas.. ]')
    if opt.get('include_personas', True):
        print(
            "\n  [NOTE: In the BST paper both partners have a persona.\n"
            + '         You can choose to ignore yours, the model never sees it.\n'
            + '         In the Blender paper, this was not used for humans.\n'
            + '         You can also turn personas off with --include-personas False]\n'
        )
    fname = raw_data_path(opt)
    with PathManager.open(fname) as json_file:
        data = json.load(json_file)
    if opt.get('include_personas', True) and opt.get('safe_personas_only', True):
        # Filter out unsafe personas
        save_personas_path = safe_personas_path(opt)
        with PathManager.open(save_personas_path, 'r') as f:
            raw_safe_persona_groups = [line.strip() for line in f.readlines()]
        safe_persona_strings = set()
        for group in raw_safe_persona_groups:
            safe_group = [_standardize(string) for string in group.split('|')]
            safe_persona_strings.update(set(safe_group))
    contexts = []
    for d in data:
        context1 = []
        context2 = []
        if opt.get('include_personas', True):
            if opt.get('safe_personas_only', True):
                personas_are_safe = all(
                    _standardize(persona_string) in safe_persona_strings
                    for persona in d['personas']
                    for persona_string in persona
                )
                if not personas_are_safe:
                    continue
            context1.append('your persona: ' + d['personas'][0][0])
            context1.append('your persona: ' + d['personas'][0][1])
            context2.append('your persona: ' + d['personas'][1][0])
            context2.append('your persona: ' + d['personas'][1][1])
        if d['context_dataset'] == 'wizard_of_wikipedia':
            context1.append(d['additional_context'])
            context2.append(d['additional_context'])
        if opt.get('include_initial_utterances', True):
            context1.append(d['free_turker_utterance'])
            context2.append(d['free_turker_utterance'])
            context1.append(d['guided_turker_utterance'])
            context2.append(d['guided_turker_utterance'])
        c1 = '\n'.join(context1)
        c2 = '\n'.join(context2)
        contexts.append([c1, c2])
    return contexts


def _standardize(orig: str) -> str:
    """
    Standardize string given punctuation differences in the list of safe personas.
    """
    new = orig.lower().rstrip('.!?')
    string_replace = {
        "i've": 'i have',
        'i ve': 'i have',
        'ive': 'i have',
        "i'm": 'i am',
        'i m': 'i am',
        'im': 'i am',
        "i'll": 'i will',
        'i ll': 'i will',
        "don't": 'do not',
        'don t': 'do not',
        'dont': 'do not',
        "can't": 'cannot',
        "can t": 'cannot',
        "cant": 'cannot',
        " s": "'s",
    }
    for i, j in string_replace.items():
        new = new.replace(i, j)
    return new


class InteractiveWorld(InteractiveBaseWorld):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('BST Interactive World')
        parser.add_argument(
            '--display-partner-persona',
            type='bool',
            default=True,
            help='Display your partner persona at the end of the chat',
        )
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )
        parser.add_argument(
            '--include-initial-utterances',
            type='bool',
            default=False,
            help='Include context conversation at beginning or not',
        )
        parser.add_argument(
            '--safe-personas-only',
            type='bool',
            default=True,
            help='Only use personas on an allowed list of safe personas',
            hidden=True,
        )
        return parser

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.display_partner_persona = self.opt['display_partner_persona']

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return p[0], p[1]

    def finalize_episode(self):
        print("\nCHAT DONE.\n")
        if self.display_partner_persona:
            partner_persona = self.p2.replace('your persona:', 'partner\'s persona:')
            print(f"Your partner was playing the following persona:\n{partner_persona}")
        if not self.epoch_done():
            print("\n[ Preparing new chat ... ]\n")

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data


from parlai.tasks.blended_skill_talk.extract import extract_from_msg
from parlai.core.worlds import validate
from parlai.core.message import Message
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
class SelfChatWorld(SelfChatBaseWorld):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('BST SelfChat World')
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )
        parser.add_argument(
            '--include-initial-utterances',
            type='bool',
            default=True,
            help='Include context conversation at beginning or not',
        )
        parser.add_argument(
            '--include-concepts',
            type='bool',
            default=False,
            help='Retrieve concepts from utterance or not',
        )
        parser.add_argument(
            '--attention-path',
            type='str',
            default="/scratch/lustre/home/illa7843/cn_extraction/self_chat/",
            help='Path where attentions should be saved',
        )
        return parser

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return [p[0], p[1]]

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data
    
    def _add_knowledge_to_act(self, act):
        text = act['text']
        concepts = extract_from_msg(text)
        knowledge = '. '.join(concepts)
        text += f'\n{TOKEN_KNOWLEDGE}{knowledge}{TOKEN_END_KNOWLEDGE}'
        act.force_set('text',text)
        return act

    def parley(self):
        if self.episode_done():
            self._end_episode()

        if self.turn_cnt == 0:
            self.acts = [None, None]
            # get any context for the beginning of the conversation
            self.contexts = self.get_contexts()

        self.seed_utterances = self._get_seed_utt_acts(self.episode_cnt, self.agents)

        if self.contexts:
            assert len(self.contexts) == 2
            # initial context
            for i in range(0, 2):
                context = Message(
                    {'text': self.contexts[i], 'episode_done': False, 'id': 'context'}
                )
                self.acts[i] = context
                self.agents[i].observe(validate(context))
            # clear contexts so they are only added once per episode
            self.contexts = None
        elif self.seed_utterances:
            # pop the next two seed messages (there may be less or more than 2 total)
            utts = self.seed_utterances[:2]
            self.seed_utterances = self.seed_utterances[2:]
            # process the turn
            for i in [0, 1]:
                # if we have a seed utterance, add it to the conversation
                if len(utts) > i:
                    self.acts[i] = utts[i]
                    if hasattr(self.agents[i], 'self_observe'):
                        self.agents[i].observe({'episode_done': False})
                        self.agents[i].self_observe(self.acts[i])
                else:
                    self.acts[i] = self.agents[i].act()
                self.agents[1 - i].observe(validate(self.acts[i]))
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents

            acts[0] = agents[0].act()            
            if self.opt.get('include_concepts', False):
                acts[0] = self._add_knowledge_to_act(acts[0])
            input_text = acts[0].get('text', '[no text field]')
            agents[1].observe(validate(acts[0]))

            acts[1] = agents[1].act()
            response_text = acts[1].get('text', 'No response')
            if self.opt.get('include_concepts', False):
                acts[1] = self._add_knowledge_to_act(acts[1])
            agents[0].observe(validate(acts[1]))

            self.save_attentions(input_text,response_text)


        self.update_counters()
        self.turn_cnt += 1

    def save_attentions(self, input_text,response_text):
        model_agent = self.get_model_agent()
        input_tokens = model_agent.dict.tokenize(input_text)
        output_tokens = model_agent.dict.tokenize(response_text)
        
        encoder = model_agent.model.encoder
        decoder = model_agent.model.decoder

        encoder_attention = [l.attention.attn_weights.unsqueeze(0) for l in encoder.layers]
        decoder_attention = [l.self_attention.attn_weights.unsqueeze(0) for l in decoder.layers]
        cross_attention = [l.encoder_attention.attn_weights.unsqueeze(0) for l in decoder.layers]
        # pad tokens
        decoder_size = decoder_attention[0].size()[-1]
        encoder_size = encoder_attention[0].size()[-1]
        output_tokens = ['__start__'] + output_tokens + ['__end__']
        output_tokens += ['__null__'] * (decoder_size - len(output_tokens))
        input_tokens += ['__null__'] * (encoder_size - len(input_tokens))
        result = {
            "input_tokens": input_tokens,
            "output_tokens":output_tokens,
            "encoder_attention":encoder_attention, 
            "decoder_attention":decoder_attention,
            "cross_attention":cross_attention
        }
        path = Path(self.opt.get('attention_path'),'./') / f'attentions_{self.total_parleys}.pickle'
        with open(path, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)        
