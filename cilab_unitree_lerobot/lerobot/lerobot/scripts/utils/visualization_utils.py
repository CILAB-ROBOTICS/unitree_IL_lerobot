import torch
import matplotlib.pyplot as plt
import wandb

def similarity_matrix(emb1, emb2):
    """
    Computes the cosine similarity matrix between two batches of embeddings.
    Returns: (Batch_Size, Batch_Size) matrix where [i, j] is sim(emb1[i], emb2[j])
    """
    emb1_norm = torch.nn.functional.normalize(emb1, dim=-1)
    emb2_norm = torch.nn.functional.normalize(emb2, dim=-1)
    return emb1_norm @ emb2_norm.T


def similarity_heatmap(sim_matrix, step, tag="default"):
    """
    Saves a heatmap of the similarity matrix.
    Returns the path to the saved plot.
    """ 
    sim_matrix_np = sim_matrix.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(sim_matrix_np, cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.xlabel(f"{tag}")
    plt.ylabel("ego")
    
    img = wandb.Image(fig)
    plt.close(fig)
    
    return img


def contrastive_loss(emb1, emb2, temperature=0.07):
    """
    Computes InfoNCE loss (CLIP-style) between two batches of embeddings.
    emb1, emb2: (Batch_Size, Dim)
    """
    # Normalize embeddings
    emb1 = torch.nn.functional.normalize(emb1, dim=-1)
    emb2 = torch.nn.functional.normalize(emb2, dim=-1)

    # Similarity matrix
    logits = torch.matmul(emb1, emb2.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    batch_size = emb1.shape[0]
    labels = torch.arange(batch_size, device=emb1.device)
    
    # Symmetric loss
    loss_1 = torch.nn.functional.cross_entropy(logits, labels)
    loss_2 = torch.nn.functional.cross_entropy(logits.T, labels)
    
    return (loss_1 + loss_2) / 2


def tsne_plot(emb1, emb2, step, output_dir, tag="default", task_indices=None, time_indices=None):
    """
    Saves a t-SNE plot of the embeddings.
    Returns the path to the saved plot.
    """
    B = emb1.size(0)
    
    # Combine embeddings
    combined = torch.cat([emb1, emb2], dim=0).detach().cpu().numpy()
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, B-1))
    reduced = tsne.fit_transform(combined)
    
    # Split back
    reduced_1 = reduced[:B]
    reduced_2 = reduced[B:]
    
    fig = plt.figure(figsize=(6, 4))
    
    # Create colors
    if task_indices is not None:
        # Ensure task_indices is a numpy array
        if isinstance(task_indices, torch.Tensor):
            task_indices = task_indices.cpu().numpy()
        if time_indices is not None and isinstance(time_indices, torch.Tensor):
            time_indices = time_indices.cpu().numpy()
        
        unique_tasks = np.unique(task_indices)
        num_tasks = len(unique_tasks)
        
        # Define a list of sequential colormaps for different tasks
        # These are single-hue or multi-hue sequential maps
        task_cmaps = [
            plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples, 
            plt.cm.Greys, plt.cm.YlOrBr, plt.cm.PuRd, plt.cm.GnBu, plt.cm.PuBuGn
        ]
        
        # Plot with legend for tasks
        for i, task in enumerate(unique_tasks):
            mask = (task_indices == task)
            task_subset_time = time_indices[mask] if time_indices is not None else None
            
            # Select colormap (cycle if more tasks than maps)
            cmap = task_cmaps[i % len(task_cmaps)]
            
            if task_subset_time is not None:
                # Normalize time for this task to 0.2-1.0 range (avoid too light colors)
                t_min, t_max = task_subset_time.min(), task_subset_time.max()
                if t_max > t_min:
                    norm_time = 0.3 + 0.7 * (task_subset_time - t_min) / (t_max - t_min)
                else:
                    norm_time = np.ones_like(task_subset_time) * 0.7
                
                colors = cmap(norm_time)
            else:
                # Solid color if no time info
                colors = cmap(0.7)
            
            # Plot ego
            plt.scatter(reduced_1[mask, 0], reduced_1[mask, 1], c=colors, marker='o', s=8, label=f'Task {task} (ego)')
            # Plot counter
            plt.scatter(reduced_2[mask, 0], reduced_2[mask, 1], c=colors, marker="*", s=8, label=f'Task {task} ({tag})')
            
    else:
        colors = cm.rainbow(np.linspace(0, 1, B))
        # Plot
        plt.scatter(reduced_1[:, 0], reduced_1[:, 1], c=colors, marker='o', s=8, label='ego')
        plt.scatter(reduced_2[:, 0], reduced_2[:, 1], c=colors, marker="*", s=8, label=f'{tag}')
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save to disk
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"tsne_{tag}_step_{step:06d}.png")
    plt.savefig(path)
    
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def accuracy(embedding1: torch.Tensor, embedding2: torch.Tensor, topk=(1, )):
    """
    Computes top-k accuracy for both directions (1->2 and 2->1).
    """
    device = embedding1.device
    B = embedding1.size(0)
    targets = torch.arange(B, device=device).view(-1, 1)

    # Cosine similarity
    emb1_norm = torch.nn.functional.normalize(embedding1, dim=-1)
    emb2_norm = torch.nn.functional.normalize(embedding2, dim=-1)
    sim_1to2 = emb1_norm @ emb2_norm.T

    metrics = {}
    max_k = max(topk)
    k_eff = min(max_k, B)

    _, topk_indices_1to2 = sim_1to2.topk(k_eff, dim=1, largest=True, sorted=True)
    _, topk_indices_2to1 = sim_1to2.t().topk(k_eff, dim=1, largest=True, sorted=True)

    for k in topk:
        if B < k:
            continue
        
        current_k = min(k, B)
        indices_1to2_k = topk_indices_1to2[:, :current_k]
        indices_2to1_k = topk_indices_2to1[:, :current_k]

        correct_1to2 = (indices_1to2_k == targets).any(dim=1)
        correct_2to1 = (indices_2to1_k == targets).any(dim=1)

        metrics[f"top{k}_1to2"] = correct_1to2.float().mean().item()
        metrics[f"top{k}_2to1"] = correct_2to1.float().mean().item()

    return metrics