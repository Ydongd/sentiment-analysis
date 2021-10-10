import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCELoss
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from .layers.linears import PoolerEndLogits, PoolerStartLogits
from .losses.crf import CRF
from .losses.focal_loss import FocalLoss
from .losses.label_smoothing import LabelSmoothingCrossEntropy
from .model_utils import valid_sequence_output


class BertSoftmax(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmax, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.activation = nn.Sigmoid()
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        pooler_output = outputs['pooler_output']
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        logits = self.activation(logits)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce', 'bce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            elif self.loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
            elif self.loss_type == 'bce':
                loss_fct = BCELoss()
            # Only keep active parts of the loss
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
