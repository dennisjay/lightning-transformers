# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from torch import nn
from transformers.models.clip.modeling_clip import clip_loss, CLIPOutput

from lightning_transformers.core.nlp import HFTransformer


class SelfSupervisedHeadModel(nn.Module):
    def __init__(self, downstream_model):
        super().__init__()
        self.downstream_model = downstream_model

        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)

        self.text_embed_dim = 768
        self.projection_dim = 512

        self.logit_scale_init_value = 2.6592

        self.logit_scale = nn.Parameter(torch.ones([]) * self.logit_scale_init_value)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_loss=True,
            **kwargs
    ):
        downstream_outputs = self.downstream_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = downstream_outputs.hidden_states
        embeddings = self._compute_embedding(hidden_states, attention_mask)

        dropout_downstream_1 = self.dropout_1(embeddings)
        dropout_downstream_2 = self.dropout_2(embeddings)

        embeds_one = dropout_downstream_1
        embeds_two = dropout_downstream_2

        # normalized features
        image_embeds = embeds_one / embeds_one.norm(dim=-1, keepdim=True)
        text_embeds = embeds_two / embeds_two.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
            logits_per_image, logits_per_text, text_embeds, image_embeds, downstream_outputs[1], downstream_outputs[1])
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=downstream_outputs[1],
            vision_model_output=downstream_outputs[1],
        )

    def _compute_embedding(
            self, hidden_states, attn_mask, layer_index=-1
    ):
        layer = hidden_states[layer_index]

        # Fix LongFormerModel like model which has mismatch seq_len between
        # attention_mask and hidden_states
        padding_len = layer.size(1) - attn_mask.size(1)
        if padding_len > 0:
            attn_mask = torch.nn.functional.pad(attn_mask, (0, padding_len), value=0)

        expand_attn_mask = attn_mask.unsqueeze(-1).expand_as(layer)

        layer = torch.where(expand_attn_mask.bool(), layer, torch.zeros_like(layer))
        embeddings = layer.sum(dim=1) / expand_attn_mask.sum(dim=1)
        return embeddings


class SelfSupervisedModelingTransformer(HFTransformer):
    """Defines ``LightningModule`` for the Language Modeling Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load. (default ``transformers.AutoModelForCausalLM``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

    def __init__(self, *args, downstream_model_type: str = "transformers.AutoModel", **kwargs) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.model = SelfSupervisedHeadModel(self.model)

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.downstream_model.resize_token_embeddings(tokenizer_length)

    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)

    @property
    def hf_pipeline_task(self) -> str:
        return "text-generation"
