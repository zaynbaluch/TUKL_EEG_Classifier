import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_tsne(features, labels, title, save_path, subsample_size=2000, perplexity=30, random_state=42):
    """
    Generate t-SNE 2D scatter plot color-coded by class.
    
    Args:
        features: np.array [N, D] — feature vectors
        labels: np.array [N] — class labels
        title: str — plot title
        save_path: str — where to save the plot
        subsample_size: int — number of samples to use (for speed)
    """
    N = len(labels)
    
    if N > subsample_size:
        indices = np.random.RandomState(random_state).choice(N, subsample_size, replace=False)
        features_sub = features[indices]
        labels_sub = labels[indices]
    else:
        features_sub = features
        labels_sub = labels

    print(f"Running t-SNE on {len(labels_sub)} samples for '{title}'...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features_sub)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels_sub,
        palette=sns.color_palette("hls", 3),
        legend="full",
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Class')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
