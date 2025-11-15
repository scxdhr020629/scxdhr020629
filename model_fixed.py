import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import warnings


class InfoNCELoss(nn.Module):
    """
    修复后的 InfoNCE 对比损失函数
    主要改进：
    1. 添加数值稳定性检查
    2. 调整温度参数的默认值
    3. 添加梯度裁剪机制
    """

    def __init__(self, temperature=0.5):  # 修改1: 温度从0.1改为0.5，避免梯度过小
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query, key):
        batch_size = query.shape[0]
        
        # 修改2: 添加数值检查
        if torch.isnan(query).any() or torch.isnan(key).any():
            print("[WARNING] NaN detected in InfoNCE inputs")
            query = torch.nan_to_num(query, nan=0.0)
            key = torch.nan_to_num(key, nan=0.0)
        
        # 修改3: 确保特征有足够的方差
        query_std = query.std(dim=1, keepdim=True)
        key_std = key.std(dim=1, keepdim=True)
        
        # 如果方差太小，说明特征坍塌了
        if query_std.mean() < 1e-6 or key_std.mean() < 1e-6:
            print(f"[WARNING] Feature collapse detected. query_std: {query_std.mean():.6f}, key_std: {key_std.mean():.6f}")
        
        # 1. 归一化特征
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)

        # 2. 计算余弦相似度
        logits = torch.matmul(query, key.T) / self.temperature
        
        # 修改4: 添加数值稳定性 - 防止logits过大
        logits = torch.clamp(logits, min=-50, max=50)

        # 3. 创建标签
        labels = torch.arange(batch_size, device=query.device)

        # 4. 计算损失
        loss_i_j = self.criterion(logits, labels)
        loss_j_i = self.criterion(logits.T, labels)

        loss = (loss_i_j + loss_j_i) / 2.0
        
        # 修改5: 添加调试信息
        with torch.no_grad():
            # 计算对角线相似度（正样本）和非对角线相似度（负样本）
            diag_sim = torch.diag(logits).mean()
            mask = torch.eye(batch_size, device=logits.device).bool()
            off_diag_sim = logits[~mask].mean()
            
            # 如果正负样本相似度没有明显差异，说明模型没有学到有用的表示
            if abs(diag_sim - off_diag_sim) < 0.1:
                print(f"[WARNING] Positive and negative similarity too close. Diag: {diag_sim:.4f}, Off-diag: {off_diag_sim:.4f}")
        
        return loss


class AttnFusionGCNNet(torch.nn.Module):
    """
    修复后的模型
    主要改进：
    1. 投影头添加BatchNorm，防止特征坍塌
    2. 投影头使用更深的结构
    3. 添加特征归一化
    4. 修复可能的梯度消失问题
    """

    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2,
                 cl_projection_dim=128):  # 修改6: 投影维度从64改为128，避免信息瓶颈

        super(AttnFusionGCNNet, self).__init__()

        self.n_output = n_output
        self.output_dim = output_dim
        self.cl_projection_dim = cl_projection_dim

        # 原有的网络层定义（保持不变）
        self.max_smile_idx = num_features_smile
        self.max_target_idx = num_features_xt
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)
        self.conv_xd_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        self.conv_xd_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)
        self.conv_xd_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=1, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=1, padding=1)
        self.fc_smiles = nn.Linear(n_filters * 2, output_dim)
        
        self.rdkit_descriptor_dim = 210
        self.rdkit_fingerprint_dim = 136
        self.maccs_dim = 166
        self.morgan_dim = 512
        self.combined_dim = 1024
        
        self.attention_rdkit_descriptor = nn.Linear(self.rdkit_descriptor_dim, self.rdkit_descriptor_dim)
        self.attention_maccs = nn.Linear(self.maccs_dim, self.maccs_dim)
        self.drug_fingerprint_transform = nn.Linear(self.combined_dim, output_dim)
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
        
        self.fusion_drug_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            self.relu,
            self.dropout
        )
        
        self.conv_reduce_smiles = nn.Conv1d(in_channels=output_dim * 3, out_channels=output_dim, kernel_size=1)
        self.conv_reduce_xt = nn.Conv1d(in_channels=192, out_channels=output_dim, kernel_size=1)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=3, padding=2)
        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters * 2, kernel_size=2, padding=1)
        
        self.conv_matrix_1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv_matrix_2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1)
        self.conv_matrix_3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1)
        self.fc_matrix_1 = nn.Linear(n_filters * 4 * 4 * 4, 256)
        self.fc_matrix_2 = nn.Linear(256, output_dim)
        
        self.fusion_mirna_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            self.relu,
            self.dropout
        )
        
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()

        # ============================================================
        # 修改7: 改进的投影头设计
        # 关键改进：
        # 1. 添加 BatchNorm 防止特征坍塌
        # 2. 使用更深的网络结构
        # 3. 不在投影头中使用 Dropout（对比学习中不推荐）
        # ============================================================
        
        # 药物分支的投影头
        self.cl_head_drug_seq = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),  # 添加BN
            self.relu,
            nn.Linear(output_dim, cl_projection_dim),
            nn.BatchNorm1d(cl_projection_dim)  # 添加BN
        )
        self.cl_head_drug_fp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            self.relu,
            nn.Linear(output_dim, cl_projection_dim),
            nn.BatchNorm1d(cl_projection_dim)
        )

        # miRNA 分支的投影头
        self.cl_head_mirna_seq = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            self.relu,
            nn.Linear(output_dim, cl_projection_dim),
            nn.BatchNorm1d(cl_projection_dim)
        )
        self.cl_head_mirna_mat = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            self.relu,
            nn.Linear(output_dim, cl_projection_dim),
            nn.BatchNorm1d(cl_projection_dim)
        )
        
        # 修改8: 添加stop gradient选项（可选）
        self.use_stop_gradient = False  # 设为True时，对比学习不会影响主干网络

    def process_drug_fingerprints(self, rdkit_descriptor, rdkit_fingerprint, maccs_fingerprint, morgan_fingerprint):
        if len(rdkit_descriptor.shape) == 1:
            rdkit_descriptor = rdkit_descriptor.unsqueeze(0)
        if len(rdkit_fingerprint.shape) == 1:
            rdkit_fingerprint = rdkit_fingerprint.unsqueeze(0)
        if len(maccs_fingerprint.shape) == 1:
            maccs_fingerprint = maccs_fingerprint.unsqueeze(0)
        if len(morgan_fingerprint.shape) == 1:
            morgan_fingerprint = morgan_fingerprint.unsqueeze(0)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if len(rdkit_descriptor.shape) > 2:
                rdkit_descriptor = rdkit_descriptor.mean(dim=1)
            if len(rdkit_fingerprint.shape) > 2:
                rdkit_fingerprint = rdkit_fingerprint.mean(dim=1)
            if len(maccs_fingerprint.shape) > 2:
                maccs_fingerprint = maccs_fingerprint.mean(dim=1)
            if len(morgan_fingerprint.shape) > 2:
                morgan_fingerprint = morgan_fingerprint.mean(dim=1)
                
        assert rdkit_descriptor.shape[-1] == self.rdkit_descriptor_dim
        assert rdkit_fingerprint.shape[-1] == self.rdkit_fingerprint_dim
        assert maccs_fingerprint.shape[-1] == self.maccs_dim
        assert morgan_fingerprint.shape[-1] == self.morgan_dim
        
        attention_weights_rdkit = self.attention_rdkit_descriptor(rdkit_descriptor)
        attention_weights_rdkit = F.softmax(attention_weights_rdkit, dim=-1)
        rdkit_descriptor_prime = rdkit_descriptor * attention_weights_rdkit
        
        attention_weights_maccs = self.attention_maccs(maccs_fingerprint)
        attention_weights_maccs = F.softmax(attention_weights_maccs, dim=-1)
        maccs_prime = maccs_fingerprint * attention_weights_maccs
        
        combined_features = torch.cat([rdkit_descriptor_prime, maccs_prime, rdkit_fingerprint, morgan_fingerprint], dim=-1)
        drug_features = self.drug_fingerprint_transform(combined_features)
        drug_features = self.relu(drug_features)
        drug_features = self.dropout(drug_features)
        drug_features = torch.nan_to_num(drug_features, nan=0.0, posinf=0.0, neginf=0.0)
        return drug_features

    def forward(self, data):
        # 数据准备
        rdkit_fingerprint = data.rdkit_fingerprint
        rdkit_descriptor = data.rdkit_descriptor
        maccs_fingerprint = data.maccs_fingerprint
        morgan_fingerprint = data.morgan_fingerprint
        drugsmile = data.seqdrug
        target = data.target
        target_matrix = data.target_matrix if hasattr(data, 'target_matrix') else None
        
        if target_matrix is None:
            raise ValueError("target_matrix is None")
            
        if drugsmile.dtype == torch.float32 or drugsmile.dtype == torch.float64:
            drugsmile = drugsmile.long()
        if target.dtype == torch.float32 or target.dtype == torch.float64:
            target = target.long()
            
        if drugsmile.max().item() > self.max_smile_idx:
            drugsmile = torch.clamp(drugsmile, 0, self.max_smile_idx)
        if drugsmile.min().item() < 0:
            drugsmile = torch.clamp(drugsmile, 0, self.max_smile_idx)
        if target.max().item() > self.max_target_idx:
            target = torch.clamp(target, 0, self.max_target_idx)
        if target.min().item() < 0:
            target = torch.clamp(target, 0, self.max_target_idx)
            
        batch_size = drugsmile.shape[0]
        
        try:
            rdkit_descriptor = rdkit_descriptor.view(batch_size, self.rdkit_descriptor_dim)
            rdkit_fingerprint = rdkit_fingerprint.view(batch_size, self.rdkit_fingerprint_dim)
            maccs_fingerprint = maccs_fingerprint.view(batch_size, self.maccs_dim)
            morgan_fingerprint = morgan_fingerprint.view(batch_size, self.morgan_dim)
        except RuntimeError as e:
            print(f"[ERROR] Reshape failed for fingerprints. Batch size: {batch_size}")
            raise e
            
        rdkit_descriptor = torch.nan_to_num(rdkit_descriptor, nan=0.0, posinf=0.0, neginf=0.0)
        rdkit_fingerprint = torch.nan_to_num(rdkit_fingerprint, nan=0.0, posinf=0.0, neginf=0.0)
        maccs_fingerprint = torch.nan_to_num(maccs_fingerprint, nan=0.0, posinf=0.0, neginf=0.0)
        morgan_fingerprint = torch.nan_to_num(morgan_fingerprint, nan=0.0, posinf=0.0, neginf=0.0)
        target_matrix = torch.nan_to_num(target_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # ============= Drug Processing =============
        fingerprint_features = self.process_drug_fingerprints(
            rdkit_descriptor=rdkit_descriptor,
            rdkit_fingerprint=rdkit_fingerprint,
            maccs_fingerprint=maccs_fingerprint,
            morgan_fingerprint=morgan_fingerprint
        )

        try:
            embedded_smile = self.smile_embed(drugsmile)
        except RuntimeError as e:
            print(f"[ERROR] smile_embed failed")
            raise e
            
        embedded_smile = embedded_smile.permute(0, 2, 1)
        conv_xd1 = self.conv_xd_11(embedded_smile)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = self.dropout(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, kernel_size=2)
        conv_xd1 = self.conv_xd_12(conv_xd1)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, conv_xd1.size(2)).squeeze(2)
        
        conv_xd2 = self.conv_xd_21(embedded_smile)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, kernel_size=2)
        conv_xd2 = self.conv_xd_22(conv_xd2)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, conv_xd2.size(2)).squeeze(2)
        
        conv_xd3 = self.conv_xd_31(embedded_smile)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = self.dropout(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, kernel_size=2)
        conv_xd3 = self.conv_xd_32(conv_xd3)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, conv_xd3.size(2)).squeeze(2)
        
        conv_xd1 = self.fc_smiles(conv_xd1)
        conv_xd2 = self.fc_smiles(conv_xd2)
        conv_xd3 = self.fc_smiles(conv_xd3)
        
        conv_xd = torch.cat((conv_xd1, conv_xd2, conv_xd3), dim=1)
        conv_xd = conv_xd.unsqueeze(1).permute(0, 2, 1)
        conv_xd = self.conv_reduce_smiles(conv_xd)
        conv_xd = conv_xd.squeeze(2)
        conv_xd = torch.nan_to_num(conv_xd, nan=0.0, posinf=0.0, neginf=0.0)

        # ============= miRNA Sequence Processing =============
        try:
            embedded_xt = self.embedding_xt(target)
        except RuntimeError as e:
            print(f"[ERROR] embedding_xt failed")
            raise e
            
        embedded_xt = embedded_xt.permute(0, 2, 1)
        conv_xt1 = self.conv_xt_11(embedded_xt)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = self.dropout(conv_xt1)
        conv_xt1 = self.conv_xt_12(conv_xt1)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = F.max_pool1d(conv_xt1, conv_xt1.size(2)).squeeze(2)
        
        conv_xt2 = self.conv_xt_21(embedded_xt)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = self.dropout(conv_xt2)
        conv_xt2 = self.conv_xt_22(conv_xt2)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = F.max_pool1d(conv_xt2, conv_xt2.size(2)).squeeze(2)
        
        conv_xt3 = self.conv_xt_31(embedded_xt)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = self.dropout(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, kernel_size=2)
        conv_xt3 = self.conv_xt_32(conv_xt3)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, conv_xt3.size(2)).squeeze(2)
        
        conv_xt = torch.cat((conv_xt1, conv_xt2, conv_xt3), dim=1)
        conv_xt = conv_xt.unsqueeze(2)
        conv_xt = self.conv_reduce_xt(conv_xt)
        conv_xt = conv_xt.squeeze(2)
        conv_xt = torch.nan_to_num(conv_xt, nan=0.0, posinf=0.0, neginf=0.0)

        # ============= miRNA Matrix Processing =============
        if len(target_matrix.shape) == 3:
            target_matrix = target_matrix.unsqueeze(1)
        matrix_feat = F.max_pool2d(self.relu(self.conv_matrix_1(target_matrix)), kernel_size=2)
        matrix_feat = F.max_pool2d(self.relu(self.conv_matrix_2(matrix_feat)), kernel_size=2)
        matrix_feat = self.dropout(self.relu(self.conv_matrix_3(matrix_feat)))
        matrix_feat = matrix_feat.view(matrix_feat.size(0), -1)
        matrix_feat = self.dropout(self.relu(self.fc_matrix_1(matrix_feat)))
        matrix_feat = self.fc_matrix_2(matrix_feat)
        matrix_feat = torch.nan_to_num(matrix_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # ============= 下游任务 (Fusion) =============
        drug_concat = torch.cat([conv_xd, fingerprint_features], dim=1)
        drug_features = self.fusion_drug_mlp(drug_concat)
        drug_features = torch.nan_to_num(drug_features, nan=0.0, posinf=0.0, neginf=0.0)

        mirna_concat = torch.cat([conv_xt, matrix_feat], dim=1)
        mirna_features = self.fusion_mirna_mlp(mirna_concat)
        mirna_features = torch.nan_to_num(mirna_features, nan=0.0, posinf=0.0, neginf=0.0)

        xc = torch.cat((drug_features, mirna_features), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.ac(out)

        # ============= 对比学习投影 =============
        # 修改9: 添加stop gradient选项
        if self.use_stop_gradient:
            conv_xd_detached = conv_xd.detach()
            fingerprint_features_detached = fingerprint_features.detach()
            conv_xt_detached = conv_xt.detach()
            matrix_feat_detached = matrix_feat.detach()
        else:
            conv_xd_detached = conv_xd
            fingerprint_features_detached = fingerprint_features
            conv_xt_detached = conv_xt
            matrix_feat_detached = matrix_feat

        cl_drug_seq = self.cl_head_drug_seq(conv_xd_detached)
        cl_drug_fp = self.cl_head_drug_fp(fingerprint_features_detached)
        cl_mirna_seq = self.cl_head_mirna_seq(conv_xt_detached)
        cl_mirna_mat = self.cl_head_mirna_mat(matrix_feat_detached)
        
        # 修改10: 添加特征统计信息（用于调试）
        if self.training and torch.rand(1).item() < 0.01:  # 1%的概率打印
            with torch.no_grad():
                print(f"\n[DEBUG] Feature Statistics:")
                print(f"  conv_xd: mean={{conv_xd.mean():.4f}}, std={{conv_xd.std():.4f}}")
                print(f"  fingerprint_features: mean={{fingerprint_features.mean():.4f}}, std={{fingerprint_features.std():.4f}}")
                print(f"  cl_drug_seq: mean={{cl_drug_seq.mean():.4f}}, std={{cl_drug_seq.std():.4f}}")
                print(f"  cl_drug_fp: mean={{cl_drug_fp.mean():.4f}}, std={{cl_drug_fp.std():.4f}}")

        return out, cl_drug_seq, cl_drug_fp, cl_mirna_seq, cl_mirna_mat
