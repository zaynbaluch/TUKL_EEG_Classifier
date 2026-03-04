import torch
import torch.nn as nn

class ConvBranch1D(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(32, output_size)

    def forward(self, x):
        # x shape: [B, C, T]
        x = self.conv(x)               # [B, 32, T//2]
        x = self.global_pool(x).squeeze(-1)  # [B, 32]
        aux = self.linear(x)          # Auxiliary output
        return x, aux

class ConvParallelEEG1DModel(nn.Module):
    def __init__(self, input_channels_list, output_size):
        super().__init__()
        print(f"Initializing ConvParallelEEG1DModel with channels: {input_channels_list}")
        self.output_size = output_size
        self.branches = nn.ModuleList([
            ConvBranch1D(in_channels, output_size)
            for in_channels in input_channels_list
        ])
        total_dim = 32 * len(input_channels_list)

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_size)
        )

    def forward(self, inputs, active_branch_indices=None):
        """
        inputs: List of tensors [B, C_i, T] for each branch
        active_branch_indices: List of indices (e.g. [0, 2]) indicating which branches to run
        """
        if active_branch_indices is None:
            active_branch_indices = list(range(len(self.branches)))  # All branches active

        main_feats = []
        aux_outputs = []

        for idx, (branch, x) in enumerate(zip(self.branches, inputs)):
            if idx in active_branch_indices:
                feat, aux = branch(x)
                # feat is [B, 32]
                main_feats.append(feat)
                aux_outputs.append(aux)
            else:
                # Append zero tensor for deactivated branches
                feat = torch.zeros(x.size(0), 32).to(x.device)
                aux = torch.zeros(x.size(0), self.output_size).to(x.device)
                main_feats.append(feat)
                aux_outputs.append(aux)

        x = torch.cat(main_feats, dim=1)  #Concat feats
        out = self.classifier(x)
        return out, aux_outputs

    def extract_features(self, inputs, active_branch_indices=None):
        """
        Extract features from each branch and the combined vector.
        Returns: feature_dict = {'branch_0': feat0, 'branch_1': feat1, ..., 'combined': combined_feat}
        """
        if active_branch_indices is None:
            active_branch_indices = list(range(len(self.branches)))

        main_feats = []
        feature_dict = {}

        for idx, (branch, x) in enumerate(zip(self.branches, inputs)):
            if idx in active_branch_indices:
                feat, _ = branch(x)
                main_feats.append(feat)
                feature_dict[f'branch_{idx}'] = feat
            else:
                feat = torch.zeros(x.size(0), 32).to(x.device)
                main_feats.append(feat)
                # Still store zeros if requested? Or skip?
                # Probably better to exclude if inactive, but let's consistency.
                feature_dict[f'branch_{idx}'] = feat

        combined_feat = torch.cat(main_feats, dim=1)
        feature_dict['combined'] = combined_feat
        return feature_dict
