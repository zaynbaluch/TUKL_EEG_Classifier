
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

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
        x = self.conv(x)               # [B, 32, T//2]
        x = self.global_pool(x).squeeze(-1)  # [B, 32]
        aux = self.linear(x)          # Auxiliary output
        return x, aux

class ConvParallelEEG1DModel(nn.Module):
    def __init__(self, input_channels_list, output_size):
        print("sahi ho bhai!")
        super().__init__()
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
                main_feats.append(feat)
                aux_outputs.append(aux)
            else:
                # Append zero tensor for deactivated branches
                feat = torch.zeros(x.size(0), 32).to(x.device)
                aux = torch.zeros(x.size(0), self.output_size).to(x.device)
                main_feats.append(feat)
                aux_outputs.append(aux)

        x = torch.cat(main_feats, dim=1)  # [B, 32 * num_active_branches]
        # x = self.fusion_gate(x) * x
        out = self.classifier(x)
        return out, aux_outputs
    
    # Standalone function to extract features without patching
def get_features_and_labels(model, loader, active_branches, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for _, x1, x2, x3, y in tqdm(loader):
            # Ensure correct shapes for Conv1d: [B, C, T]
            x1 = x1.to(device).unsqueeze(1) if x1.dim() == 2 else x1.to(device)  # Raw EEG: [B, 1, 400]
            x2 = x2.to(device) if x2.dim() == 3 else x2.to(device).unsqueeze(0)  # Power spectrum: [B, 33, 14]
            x3 = x3.to(device) if x3.dim() == 3 else x3.to(device).unsqueeze(0)  # Wavelet: [B, 30, 400]
            inputs = [x1, x2, x3]
            
            main_feats = []
            for idx, (branch, x) in enumerate(zip(model.branches, inputs)):
                if idx in active_branches:
                    feat, _ = branch(x)
                    main_feats.append(feat)
                else:
                    feat = torch.zeros(x.size(0), 32).to(x.device)
                    main_feats.append(feat)
            
            feat = torch.cat(main_feats, dim=1)  # [B, 96]
            features.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())   
    return np.vstack(features), np.hstack(labels)

# # Usage example (integrate into your notebook after model and val_loader)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# active_branches = [0]  # From your code
# features, labels = get_features_and_labels(model, val_loader, active_branches, device)

# # Optional: Normalize for fair distance comparison
# features = features / np.linalg.norm(features, axis=1, keepdims=True)  # Unit vectors

# # Group by class (assuming classes 0, 1, 2)
# class_feats = {c: features[labels == c] for c in [0, 1, 2] if np.any(labels == c)}

# # Compute centroids and distances (Euclidean)
# centroids = {c: np.mean(feats, axis=0) for c, feats in class_feats.items() if len(feats) > 0}
# intra_distances = {c: np.mean(np.linalg.norm(class_feats[c] - centroids[c], axis=1)) for c in centroids if len(class_feats[c]) > 1}
# inter_distances = {}
# for c1 in centroids:
#     for c2 in centroids:
#         if c1 < c2:
#             inter_distances[f"{c1}-{c2}"] = np.linalg.norm(centroids[c1] - centroids[c2])

# print("Intra-Class Distances:", intra_distances)
# print("Inter-Class Distances:", inter_distances)