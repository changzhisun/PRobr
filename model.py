from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils import get_device_of

class RobertaForRR(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRR, self).__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None,
                head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (qa_loss,) + outputs

        return outputs  # qa_loss, logits, (hidden_states), (attentions)

class RobertaForRRWithNodeLoss(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRRWithNodeLoss, self).__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.naf_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier_node = NodeClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, proof_offset=None, node_label=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        naf_output = self.naf_layer(cls_output)
        logits = self.classifier(sequence_output)

        max_node_length = node_label.shape[1]
        batch_size = node_label.shape[0]
        embedding_dim = sequence_output.shape[2]
        device = get_device_of(input_ids)

        batch_node_embedding = torch.zeros((batch_size, max_node_length, embedding_dim)).to(device)
        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            count = 0
            for offset in proof_offset[batch_index]:
                if offset == 0:
                    break
                else:
                    rf_embedding = torch.mean(sequence_output[batch_index, prev_index:(offset+1), :], dim=0).unsqueeze(0)
                    prev_index = offset+1
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = rf_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)

            # Add the NAF output at the end
            sample_node_embedding = torch.cat((sample_node_embedding, naf_output[batch_index].unsqueeze(0)), dim=0)

            # Append 0s at the end (these will be ignored for loss)
            sample_node_embedding = torch.cat((sample_node_embedding, torch.zeros((max_node_length-count-1, embedding_dim)).to(device)), dim=0)
            batch_node_embedding[batch_index, :, :] = sample_node_embedding

        node_logits = self.classifier_node(batch_node_embedding)

        outputs = (logits, node_logits) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            node_loss = loss_fct(node_logits.view(-1, self.num_labels), node_label.view(-1))
            total_loss = qa_loss + node_loss
            outputs = (total_loss, qa_loss, node_loss) + outputs

        return outputs  # (total_loss), qa_loss, node_loss, logits, node_logits, (hidden_states), (attentions)

class NodeClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, latent=False):
        super(NodeClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if latent:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels * 2)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EdgeClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, latent=False):
        super(EdgeClassificationHead, self).__init__()
        self.dense = nn.Linear(3*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if latent:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels * 2 * 2 * 2)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForRRWithNodeEdgeLoss(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRRWithNodeEdgeLoss, self).__init__(config)

        self.num_labels = config.num_labels
        self.num_labels_edge = 2
        self.roberta = RobertaModel(config)
        self.naf_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = RobertaClassificationHead(config)
        self.classifier_node = NodeClassificationHead(config)
        self.classifier_edge = EdgeClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, proof_offset=None, node_label=None,
                edge_label=None, labels=None, position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        naf_output = self.naf_layer(cls_output)
        logits = self.classifier(sequence_output)

        max_node_length = node_label.shape[1]
        max_edge_length = edge_label.shape[1]
        batch_size = node_label.shape[0]
        embedding_dim = sequence_output.shape[2]
        device = get_device_of(input_ids)

        batch_node_embedding = torch.zeros((batch_size, max_node_length, embedding_dim)).to(device)
        batch_edge_embedding = torch.zeros((batch_size, max_edge_length, 3*embedding_dim)).to(device)

        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            count = 0
            for offset in proof_offset[batch_index]:
                if offset == 0:
                    break
                else:
                    rf_embedding = torch.mean(sequence_output[batch_index, prev_index:(offset+1), :], dim=0).unsqueeze(0)
                    prev_index = offset+1
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = rf_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)

            # Add the NAF output at the end
            sample_node_embedding = torch.cat((sample_node_embedding, naf_output[batch_index].unsqueeze(0)), dim=0)

            repeat1 = sample_node_embedding.unsqueeze(0).repeat(len(sample_node_embedding), 1, 1)
            repeat2 = sample_node_embedding.unsqueeze(1).repeat(1, len(sample_node_embedding), 1)
            sample_edge_embedding = torch.cat((repeat1, repeat2, (repeat1-repeat2)), dim=2)

            sample_edge_embedding = sample_edge_embedding.view(-1, sample_edge_embedding.shape[-1])

            # Append 0s at the end (these will be ignored for loss)
            sample_node_embedding = torch.cat((sample_node_embedding,
                                               torch.zeros((max_node_length-count-1, embedding_dim)).to(device)), dim=0)
            sample_edge_embedding = torch.cat((sample_edge_embedding,
                                               torch.zeros((max_edge_length-len(sample_edge_embedding), 3*embedding_dim)).to(device)), dim=0)

            batch_node_embedding[batch_index, :, :] = sample_node_embedding
            batch_edge_embedding[batch_index, :, :] = sample_edge_embedding

        node_logits = self.classifier_node(batch_node_embedding)
        edge_logits = self.classifier_edge(batch_edge_embedding)

        outputs = (logits, node_logits, edge_logits) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            node_loss = loss_fct(node_logits.view(-1, self.num_labels), node_label.masked_fill(node_label < 0, -100).view(-1))
            edge_loss = loss_fct(edge_logits.view(-1, self.num_labels_edge), edge_label.masked_fill(edge_label < 0, -100).view(-1))
            total_loss = qa_loss + node_loss + edge_loss
            outputs = (total_loss, qa_loss, node_loss, edge_loss) + outputs

        return outputs  # (total_loss), qa_loss, node_loss, edge_loss, logits, node_logits, edge_logits, (hidden_states), (attentions)

class MyModel(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(MyModel, self).__init__(config)

        self.num_labels = config.num_labels
        self.num_labels_edge = 2
        self.roberta = RobertaModel(config)
        self.naf_layer = nn.Linear(config.hidden_size, config.hidden_size)
        # variational parameters
        self.classifier = RobertaClassificationHead(config)
        self.classifier_node = NodeClassificationHead(config)
        self.classifier_edge = EdgeClassificationHead(config)

        # joint parameters
        self.classifier_qa = RobertaClassificationHead(config)
        self.classifier_node_qa = NodeClassificationHead(config, True)
        self.classifier_node12_edge_qa = EdgeClassificationHead(config, True)

        self.apply(self.init_weights)
        self.loss_fct = CrossEntropyLoss()
        self.kl_loss_fct = nn.KLDivLoss(reduction="none")
        self.t = 1.0

    def masked_kl_loss(self, mask, logits, posterior_logits):
        eps = 0.000001
        log_softmax = F.log_softmax(logits / self.t, dim=-1) 
        posterior_softmax = F.softmax(posterior_logits / self.t, dim=-1) 
        loss_matrix = self.kl_loss_fct(log_softmax, posterior_softmax)
        loss_matrix = loss_matrix * mask.unsqueeze(-1)
        loss = loss_matrix.sum() / (mask.sum() + eps)
        return loss * self.t * self.t
    
    def compute_embeddings(self, sequence_output, proof_offset, node_label, edge_label):
        cls_output = sequence_output[:, 0, :]
        naf_output = self.naf_layer(cls_output)

        max_node_length = node_label.shape[1]
        max_edge_length = edge_label.shape[1]
        batch_size = node_label.shape[0]
        embedding_dim = sequence_output.shape[2]
        device = get_device_of(sequence_output)

        batch_node_embedding = torch.zeros((batch_size, max_node_length, embedding_dim)).to(device)
        batch_edge_embedding = torch.zeros((batch_size, max_edge_length, 3*embedding_dim)).to(device)

        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            count = 0
            for offset in proof_offset[batch_index]:
                if offset == 0:
                    break
                else:
                    rf_embedding = torch.mean(sequence_output[batch_index, prev_index:(offset+1), :], dim=0).unsqueeze(0)
                    prev_index = offset+1
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = rf_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)

            # Add the NAF output at the end
            sample_node_embedding = torch.cat((sample_node_embedding, naf_output[batch_index].unsqueeze(0)), dim=0)

            repeat1 = sample_node_embedding.unsqueeze(0).repeat(len(sample_node_embedding), 1, 1)
            repeat2 = sample_node_embedding.unsqueeze(1).repeat(1, len(sample_node_embedding), 1)
            sample_edge_embedding = torch.cat((repeat1, repeat2, (repeat1-repeat2)), dim=2)

            sample_edge_embedding = sample_edge_embedding.view(-1, sample_edge_embedding.shape[-1])

            # Append 0s at the end (these will be ignored for loss)
            sample_node_embedding = torch.cat((sample_node_embedding,
                                               torch.zeros((max_node_length-count-1, embedding_dim)).to(device)), dim=0)
            sample_edge_embedding = torch.cat((sample_edge_embedding,
                                               torch.zeros((max_edge_length-len(sample_edge_embedding), 3*embedding_dim)).to(device)), dim=0)

            batch_node_embedding[batch_index, :, :] = sample_node_embedding
            batch_edge_embedding[batch_index, :, :] = sample_edge_embedding
        return batch_node_embedding, batch_edge_embedding

    def compute_posterior_qa(self, qa_logits, node_qa_logits, node12_edge_qa_logits, node_label, edge_label, node_lengths):
        max_node_length = node_qa_logits.size(1)
        max_edge_length = node12_edge_qa_logits.size(1)
        batch_size = node_qa_logits.size(0)

        node_qa_logits = node_qa_logits.view(-1, max_node_length, 2, 2)
        qa_logits_1 = torch.gather(
            node_qa_logits,
            2,
            node_label.view(batch_size, max_node_length, 1, 1).expand(batch_size, max_node_length, 1, 2),
        ).squeeze(2).sum(1)

        node12_edge_qa_logits = node12_edge_qa_logits.view(-1, max_edge_length, 2, 2, 2, 2)
        node12_qa_logits = torch.gather(
            node12_edge_qa_logits,
            4,
            edge_label.view(batch_size, max_edge_length, 1, 1, 1, 1).expand(batch_size, max_edge_length, 2, 2, 1, 2),
        ).squeeze(4)
        batch_qa_logits = []
        for batch_idx, node_length in enumerate(node_lengths):
            cur_edge_length = node_length * node_length
            cur_node_label = node_label[batch_idx][:node_length]
            # node_length x node_length x 2 x 2
            cur_node12_qa_logits = node12_qa_logits[batch_idx][:cur_edge_length].view(node_length, node_length, 2, 2, 2)
            cur_qa_logits = cur_node12_qa_logits.sum(0).sum(0).sum(0).sum(0)
            batch_qa_logits.append(cur_qa_logits)
        qa_logits_2 = torch.stack(batch_qa_logits)
        return qa_logits_1 + qa_logits_2 + qa_logits

    def compute_posterior_node(self, node_qa_logits, node12_edge_qa_logits, edge_label, labels, node_lengths):
        max_node_length = node_qa_logits.size(1)
        max_edge_length = node12_edge_qa_logits.size(1)
        batch_size = node_qa_logits.size(0)

        device = get_device_of(node_qa_logits)
        node_qa_logits = node_qa_logits.view(-1, max_node_length, 2, 2)
        node12_edge_qa_logits = node12_edge_qa_logits.view(-1, max_edge_length, 2, 2, 2, 2)
        node_logits_1 = torch.gather(
            node_qa_logits,
            3,
            labels.view(batch_size, 1, 1, 1).expand(batch_size, max_node_length, 2, 1),
        ).squeeze(-1)

        node12_edge_logits = torch.gather(
            node12_edge_qa_logits,
            5,
            labels.view(batch_size, 1, 1, 1, 1, 1).expand(batch_size, max_edge_length, 2, 2, 2, 1),
        ).squeeze(-1)
        node12_logits = torch.gather(
            node12_edge_logits,
            4,
            edge_label.view(batch_size, max_edge_length, 1, 1, 1).expand(batch_size, max_edge_length, 2, 2, 1),
        ).squeeze(-1)
        batch_node_logits = []
        for batch_idx, node_length in enumerate(node_lengths):
            cur_edge_length = node_length * node_length
            # node_length x node_length x 2 x 2
            cur_node12_logits = node12_logits[batch_idx][:cur_edge_length].view(node_length, node_length, 2, 2)
            cur_node1_logits = cur_node12_logits.sum(3).sum(1) # (node_length x node_length x 2) -> (node_length x 2)
            cur_node2_logits = cur_node12_logits.sum(2).sum(0) # (node_length x node_length x 2) -> (ndoe_length x 2)
            cur_node_logits = cur_node1_logits + cur_node2_logits # (node_length x 2)
            cur_node_logits = torch.cat([cur_node_logits, torch.zeros(max_node_length - node_length, 2).to(device)], 0)
            batch_node_logits.append(cur_node_logits)
        node_logits_2 = torch.stack(batch_node_logits)
        node_logits = node_logits_1 + node_logits_2
        return node_logits

    def compute_posterior_edge(self, node12_edge_qa_logits, labels):
        max_edge_length = node12_edge_qa_logits.size(1)
        batch_size = labels.size(0)
        # (batch_size, max_edge_length, node1, node2, edge, qa)
        node12_edge_qa_logits = node12_edge_qa_logits.view(-1, max_edge_length, 2, 2, 2, 2)
        node12_edge_logits = torch.gather(
            node12_edge_qa_logits,
            5,
            labels.view(batch_size, 1, 1, 1, 1, 1).expand(batch_size, max_edge_length, 2, 2, 2, 1),
        ).squeeze(-1)
        edge_logits = node12_edge_logits.sum(2).sum(2) # batch_size x max_edge_length x 2
        return edge_logits

    def compute_qa_node_edge_loss(self, qa_logits, node_logits, edge_logits, labels, node_label, edge_label):
        qa_loss = self.loss_fct(qa_logits.view(-1, self.num_labels), labels.masked_fill(labels < 0, -100).view(-1))
        node_loss = self.loss_fct(node_logits.view(-1, self.num_labels), node_label.masked_fill(node_label < 0, -100).view(-1))
        edge_loss = self.loss_fct(edge_logits.view(-1, self.num_labels_edge), edge_label.masked_fill(edge_label < 0, -100).view(-1))
        return qa_loss, node_loss, edge_loss

    def compute_kl_loss(self, 
            variational_qa_logits, variational_node_logits, variational_edge_logits,
            posterior_qa_logits, posterior_node_logits, posterior_edge_logits,
            labels, node_label, edge_label,
        ):
        #  qa_mask = (labels != -100)
        #  node_mask = (node_label != -100)
        #  edge_mask = (edge_label != -100)
        qa_mask = (labels == -1) + (labels == -2)
        node_mask = (node_label == -1) + (node_label == -2)
        edge_mask = (edge_label == -1) + (edge_label == -2)
        qa_loss = self.masked_kl_loss(
            qa_mask.view(-1),
            variational_qa_logits, 
            posterior_qa_logits,
        )
        node_loss = self.masked_kl_loss(
            node_mask.view(-1),
            variational_node_logits.view(-1, self.num_labels), 
            posterior_node_logits.view(-1, self.num_labels),
        )
        edge_loss = self.masked_kl_loss(
            edge_mask.view(-1),
            variational_edge_logits.view(-1, self.num_labels_edge), 
            posterior_edge_logits.view(-1, self.num_labels_edge),
        )
        return qa_loss, node_loss, edge_loss

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, proof_offset=None, node_label=None,
                edge_label=None, labels=None, position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        node_lengths = (proof_offset != 0).sum(1) + 1
        batch_node_embedding, batch_edge_embedding = self.compute_embeddings(sequence_output, proof_offset, node_label, edge_label)
        batch_size = batch_node_embedding.size(0)
        max_node_length = batch_node_embedding.size(1)
        max_edge_length = batch_edge_embedding.size(1)

        variational_qa_logits = self.classifier(sequence_output)               # batch_size x 2
        variational_node_logits = self.classifier_node(batch_node_embedding)   # batch_size x max_node_length x 2
        variational_edge_logits = self.classifier_edge(batch_edge_embedding)   # batch_size x max_edge_length x 2
        
        variational_node_logits = variational_node_logits * ((node_label != -100).view(batch_size, max_node_length, 1))
        variational_edge_logits = variational_edge_logits * ((edge_label != -100).view(batch_size, max_edge_length, 1))

        variational_qa_label = variational_qa_logits.argmax(-1)
        variational_node_label = variational_node_logits.argmax(-1)
        variational_edge_label = variational_edge_logits.argmax(-1)

        qa_logits = self.classifier_qa(sequence_output)
        node_qa_logits = self.classifier_node_qa(batch_node_embedding) # batch_size x max_node_length x 4
        node_qa_logits = node_qa_logits * ((node_label != -100).view(batch_size, max_node_length, 1))

        node12_edge_qa_logits = self.classifier_node12_edge_qa(batch_edge_embedding) # batch_size x max_edge_length x 16
        node12_edge_qa_logits = node12_edge_qa_logits * ((edge_label != -100).view(batch_size, max_edge_length, 1))

        #  cond_node_label = node_label.masked_fill(node_label < 0, 0) if self.training else variational_node_label
        #  cond_edge_label = edge_label.masked_fill(edge_label < 0, 0) if self.training else variational_edge_label
        cond_node_label = variational_node_label
        cond_edge_label = variational_edge_label
        cond_qa_label = labels.masked_fill(labels < 0, 0) if self.training else variational_qa_label
        posterior_node_logits = self.compute_posterior_node(
            node_qa_logits, node12_edge_qa_logits, 
            cond_edge_label, cond_qa_label, 
            node_lengths,
        )
        posterior_edge_logits = self.compute_posterior_edge(
            node12_edge_qa_logits, cond_qa_label,
        )
        posterior_qa_logits = self.compute_posterior_qa(
            qa_logits, node_qa_logits, node12_edge_qa_logits, 
            cond_node_label, cond_edge_label,
            node_lengths,
        )
        variational_qa_loss, variational_node_loss, variational_edge_loss = self.compute_qa_node_edge_loss(
            variational_qa_logits, variational_node_logits, variational_edge_logits,
            labels, node_label, edge_label,
        )

        posterior_qa_loss, posterior_node_loss, posterior_edge_loss = self.compute_qa_node_edge_loss( 
            posterior_qa_logits, posterior_node_logits, posterior_edge_logits,
            labels, node_label, edge_label,
        )

        kl_qa_loss, kl_node_loss, kl_edge_loss = self.compute_kl_loss(
            variational_qa_logits, variational_node_logits, variational_edge_logits,
            posterior_qa_logits, posterior_node_logits, posterior_edge_logits,
            labels, node_label, edge_label,
        )
        
        qa_loss = posterior_qa_loss + kl_qa_loss
        node_loss = variational_node_loss + kl_node_loss
        edge_loss = variational_edge_loss + kl_edge_loss

        #  qa_loss = posterior_qa_loss + kl_qa_loss
        #  node_loss = posterior_node_loss + kl_node_loss
        #  edge_loss = posterior_edge_loss + kl_edge_loss

        logits, node_logits, edge_logits = posterior_qa_logits, variational_node_logits, variational_edge_logits
        #  logits, node_logits, edge_logits = variational_qa_logits, variational_node_logits, variational_edge_logits
        #  logits, node_logits, edge_logits = posterior_qa_logits, posterior_node_logits, posterior_edge_logits
        #  (logits, node_logits, edge_logits) = (
            #  variational_qa_logits + posterior_qa_logits, 
            #  variational_node_logits + posterior_node_logits, 
            #  variational_edge_logits + posterior_edge_logits,
        #  )
        #  print("Variational: QA Loss = %.4f, Node Loss = %.4f, Edge Loss = %.4f" %
                    #  (variational_qa_loss, variational_node_loss, variational_edge_loss))
        #  print("Posterior: QA Loss = %.4f, Node Loss = %.4f, Edge Loss = %.4f" %
                    #  (posterior_qa_loss, posterior_node_loss, posterior_edge_loss))
        #  print("KL: QA Loss = %.4f, Node Loss = %.4f, Edge Loss = %.4f" %
                    #  (kl_qa_loss, kl_node_loss, kl_edge_loss))
        outputs = (logits, node_logits, edge_logits) + outputs[2:]
        total_loss = qa_loss + node_loss + edge_loss
        outputs = (total_loss, qa_loss, node_loss, edge_loss) + outputs
        return outputs  # (total_loss), qa_loss, node_loss, edge_loss, logits, node_logits, edge_logits, (hidden_states), (attentions)
