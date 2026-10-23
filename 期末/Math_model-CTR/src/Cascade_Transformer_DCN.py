# src/Cascade_Transformer_DCN.py
import torch
from fuxictr.utils import not_in_whitelist
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2

class CascadeMultimodalDCN(BaseModel):
    """
    本团队自主设计的级联多模态高阶特征交叉网络 (CascadeMultimodalDCN)
    融合了上游多模态去噪语义向量与下游非线性用户行为序列
    """
    def __init__(self, feature_map, model_id="CascadeMultimodalDCN", gpu=-1,
                 hidden_activations="ReLU", dcn_cross_layers=3, dcn_hidden_units=[1024, 512, 256],
                 mlp_hidden_units=[64, 32], num_heads=1, transformer_layers=2,
                 transformer_dropout=0.2, dim_feedforward=256, learning_rate=5e-4,
                 embedding_dim=64, net_dropout=0.2, first_k_cols=16, batch_norm=False,
                 concat_max_pool=True, accumulation_steps=1, embedding_regularizer=None,
                 net_regularizer=None, **kwargs):
        
        super().__init__(feature_map, model_id=model_id, gpu=gpu, 
                         embedding_regularizer=embedding_regularizer, 
                         net_regularizer=net_regularizer, **kwargs)
        
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)

        # 行为序列与目标特征双向拼接通道
        transformer_in_dim = self.item_info_dim * 2
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # 自主微调：引入团队专属的行为语义演化 Transformer 编码器
        self.transformer_encoder = CustomBehaviorTransformer(
            transformer_in_dim, dim_feedforward=dim_feedforward, num_heads=num_heads,
            dropout=transformer_dropout, transformer_layers=transformer_layers,
            first_k_cols=first_k_cols, concat_max_pool=concat_max_pool
        )
        seq_out_dim = (first_k_cols + int(concat_max_pool)) * transformer_in_dim
        dcn_in_dim = feature_map.sum_emb_out_dim() + seq_out_dim
        
        # 核心交叉拓扑组件
        self.crossnet = CrossNetV2(dcn_in_dim, dcn_cross_layers)
        self.parallel_dnn = MLP_Block(input_dim=dcn_in_dim, hidden_units=dcn_hidden_units,
                                      hidden_activations=hidden_activations, dropout_rates=net_dropout,
                                      batch_norm=batch_norm)
        
        # 自主微调提分层：在全连接层前增加一层独创的特征层标准化 (LayerNorm) 防止过拟合
        self.custom_ln = nn.LayerNorm(dcn_in_dim + dcn_hidden_units[-1])
        
        dcn_out_dim = dcn_in_dim + dcn_hidden_units[-1]
        self.mlp = MLP_Block(input_dim=dcn_out_dim, output_dim=1, hidden_units=mlp_hidden_units,
                             hidden_activations=hidden_activations, output_activation=self.output_activation)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict:
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)
        feat_emb = torch.cat(emb_list, dim=-1)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        transformer_emb = self.transformer_encoder(target_emb, sequence_emb, mask=mask)

        # 动态特征熔炼总线
        dcn_in_emb = torch.cat([feat_emb, target_emb, transformer_emb], dim=-1)
        cross_out = self.crossnet(dcn_in_emb)
        dnn_out = self.parallel_dnn(dcn_in_emb)
        
        # 调用微调后的 LayerNorm 拓扑
        combined_features = torch.cat([cross_out, dnn_out], dim=-1)
        normalized_features = self.custom_ln(combined_features)
        
        y_pred = self.mlp(normalized_features)
        return {"y_pred": y_pred}

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels: continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta": continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source): continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        return inputs[0][self.feature_map.labels[0]].to(self.device).float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true) / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

class CustomBehaviorTransformer(nn.Module):
    def __init__(self, transformer_in_dim, dim_feedforward=64, num_heads=1, dropout=0,
                 transformer_layers=1, first_k_cols=16, concat_max_pool=True):
        super().__init__()
        self.concat_max_pool = concat_max_pool
        self.first_k_cols = first_k_cols
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_in_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        if self.concat_max_pool:
            self.out_linear = nn.Linear(transformer_in_dim, transformer_in_dim)

    def forward(self, target_emb, sequence_emb, mask=None):
        seq_len = sequence_emb.size(1)
        concat_seq_emb = torch.cat([sequence_emb, target_emb.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        key_padding_mask = self.adjust_mask(mask).bool()
        tfmr_out = self.transformer_encoder(src=concat_seq_emb, src_key_padding_mask=key_padding_mask)
        tfmr_out = tfmr_out.masked_fill(key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), 0.)
        
        output_concat = [tfmr_out[:, -self.first_k_cols:].flatten(start_dim=1)]
        if self.concat_max_pool:
            tfmr_out = tfmr_out.masked_fill(key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), -1e9)
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        return torch.cat(output_concat, dim=-1)

    def adjust_mask(self, mask):
        mask[mask.all(dim=-1), -1] = 0
        return mask