import torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F

class transformer_block(nn.Module):
    def __init__(self, in_dim, hidden_dim, head=1):
        super(transformer_block, self).__init__()
        self.head = head

        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        self.concat_trans = nn.Linear((hidden_dim)*head, hidden_dim, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.layernorm = nn.LayerNorm(in_dim)
    
    def forward(self, g, residue_h, return_att=False):
        multi_output = []
        att_list = [] 
        for i in range(self.head):
            
            q = self.trans_q_list[i](residue_h)
            k = self.trans_k_list[i](residue_h)
            v = self.trans_v_list[i](residue_h)
            att = torch.sum(torch.mul(q, k)/torch.sqrt(torch.tensor(1280.0)), dim=1, keepdim=True)

            with g.local_scope():
                g.ndata['att'] = att.reshape(-1)
                alpha = dgl.softmax_nodes(g, 'att').reshape((v.size(0), 1))
                tp = v * alpha
                if return_att:
                    att_list.append(alpha) 
            multi_output.append(tp)

        multi_output = torch.cat(multi_output, dim=1)
        multi_output = self.concat_trans(multi_output)

        multi_output = self.layernorm(multi_output + residue_h)

        multi_output = self.layernorm(self.ff(multi_output)+multi_output)
        if return_att:
            avg_att = torch.mean(torch.stack(att_list, dim=0), dim=0)
            return multi_output, avg_att
        return multi_output

class GCN_Parallel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, head, dropout=0.3):
        super(GCN_Parallel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gcn_layers = nn.ModuleList([
            dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
            dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        ])
        self.bn_gcn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(2)])
        self.gat_layers = nn.ModuleList([
            dglnn.GATConv(in_dim, hidden_dim // head, num_heads=head, allow_zero_in_degree=True),
            dglnn.GATConv(hidden_dim, hidden_dim // head, num_heads=head, allow_zero_in_degree=True)
        ])
        self.bn_gat = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(2)])
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()

        )
        self.transformer_block = transformer_block(hidden_dim, hidden_dim, head)

    def forward(self, g, h,return_node_feat=False, return_att=False):
        with g.local_scope():
            g.ndata['h'] = h
            init_avg_h = dgl.mean_nodes(g, 'h')
            # --- GCN Forward ---
            h_gcn = h
            for i, layer in enumerate(self.gcn_layers):
                h_res = h_gcn
                h_gcn = layer(g, h_gcn)
                h_gcn = self.bn_gcn[i](h_gcn)
                h_gcn = F.relu(h_gcn)
                h_gcn = self.dropout(h_gcn)
                if i > 0: h_gcn = h_gcn + h_res
            # --- GAT Forward ---
            h_gat = h
            for i, layer in enumerate(self.gat_layers):
                h_res = h_gat
                h_gat = layer(g, h_gat).flatten(1)
                h_gat = self.bn_gat[i](h_gat)
                h_gat = F.elu(h_gat)
                h_gat = self.dropout(h_gat)
                if i > 0: h_gat = h_gat + h_res

            cat_feat = torch.cat([h_gcn, h_gat], dim=1) # [N, 2*Hid]
            gate = self.gate_fc(cat_feat)               # [N, Hid] (0~1)
            h_fused = gate * h_gcn + (1 - gate) * h_gat
            residue_h = h_fused
            if return_att:
                hg, att_weights = self.transformer_block(g, residue_h, return_att=True)
                
                return hg, att_weights 
            else:
                hg = self.transformer_block(g, residue_h, return_att=False)
            if return_node_feat:
                return hg, init_avg_h
            g.ndata['output'] = hg
            readout = dgl.sum_nodes(g, "output")
            return readout, init_avg_h

class MultiScaleMotifBlock(nn.Module):
    def __init__(self, in_channels=1280, out_channels=512):
        super(MultiScaleMotifBlock, self).__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv9 = nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4)
        self.conv17 = nn.Conv1d(out_channels, out_channels, kernel_size=17, padding=8)
        self.dilated = nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        
        self.fusion = nn.Conv1d(out_channels * 4, out_channels, kernel_size=1)
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels, kernel_size=1), 
            nn.Sigmoid()
        )


    def forward(self, x):
        # x: [Batch, Length, Dim] -> [B, Dim, L]
        x = x.permute(0, 2, 1)
        x = F.relu(self.proj(x))
        
        f3 = F.relu(self.conv3(x))
        f9 = F.relu(self.conv9(x))
        f17 = F.relu(self.conv17(x))
        fd = F.relu(self.dilated(x))
        
        cat = torch.cat([f3, f9, f17, fd], dim=1)
        out = self.fusion(cat)
        
        w = self.gate(out) 
        out = out * w
       
        out = out + x
        
        return out.permute(0, 2, 1)

class SSAS_Net(nn.Module):
    def __init__(self, graph_size, graph_hid, label_num, head):
        super(SSAS_Net, self).__init__()
        
        hidden_dim = graph_hid
        
        self.struct_encoder = GCN_Parallel(graph_size, hidden_dim, label_num, head=head)
        
        self.seq_encoder = MultiScaleMotifBlock(in_channels=graph_size, out_channels=hidden_dim)
        
        self.modal_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, label_num)
        )

    def forward(self, g, h_esm,return_att=False):
        if return_att:
            _, att_weights = self.struct_encoder(g, h_esm, return_att=True)
            return att_weights
        
        h_struct, _ = self.struct_encoder(g, h_esm, return_node_feat=True) # [N, 512]
        
       
        h_seq = self.seq_encoder(h_esm.unsqueeze(0)).squeeze(0) # [N, 512]
        
        
        cat_feat = torch.cat([h_struct, h_seq], dim=-1)
        
        
        alpha = self.modal_gate(cat_feat)
        
        
        h_fused = alpha * h_struct + (1 - alpha) * h_seq
        
        
        with g.local_scope():
            g.ndata['h'] = h_fused
            readout = dgl.sum_nodes(g, 'h')
            
        return self.classifier(readout)
