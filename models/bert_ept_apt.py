from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn

from utils.tools import get_feature, Pooling


class Bert_ept_apt(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.pooling = Pooling(word_embedding_dimension=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            input_ids1=None,
            attention_mask1=None,
            token_type_ids1=None,
            alpha=1.0,
            beta=1.0,
            ap_layer=5
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        output_attentions = True
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        if input_ids1 is not None and attention_mask1 is not None:
            outputs1 = self.bert(
                input_ids=input_ids1,
                attention_mask=attention_mask1,
                token_type_ids=token_type_ids1,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            features1 = get_feature(outputs['hidden_states'][0], outputs['pooler_output'], attention_mask)
            mean_pooling1 = self.pooling(features1)['sentence_embedding']
            features2 = get_feature(outputs1['hidden_states'][0], outputs1['pooler_output'], attention_mask1)
            mean_pooling2 = self.pooling(features2)['sentence_embedding']
            ep_loss = - MSELoss()(mean_pooling2, mean_pooling1)

            pooled_output1 = outputs1[1]
            pooled_output1 = self.dropout(pooled_output1)
            logits1 = self.classifier(pooled_output1)
            rb_loss = CrossEntropyLoss()(logits1.view(-1, self.num_labels), labels.view(-1))

            attentions = outputs['attentions'][ap_layer]
            attentions1 = outputs1['attentions'][ap_layer]
            ap_loss = MSELoss()(attentions, attentions1)

            return {
                'loss': loss + alpha * ep_loss + rb_loss + beta * ap_loss,  # + loss3,
                'logits': logits,
                'hidden_states': outputs.hidden_states,
                'attentions': outputs.attentions,
            }

        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

