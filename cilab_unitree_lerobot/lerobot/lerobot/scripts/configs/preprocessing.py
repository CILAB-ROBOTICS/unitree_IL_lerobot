import torch


def train_test_split(dataset, val_ratio=0.2, seed=0):
    epi = dataset.episode_data_index
    lengths = [
        (epi["to"][i] - epi["from"][i]).item()
        for i in range(len(epi["from"]))
    ]

    episodes = list(range(len(lengths)))

    g = torch.Generator().manual_seed(seed)
    episodes = torch.tensor(episodes)[torch.randperm(len(episodes), generator=g)].tolist()

    total_frames = sum(lengths)
    val_target = total_frames * val_ratio

    val_eps, train_eps = [], []
    acc = 0
    for ep in episodes:
        if acc < val_target:
            val_eps.append(ep)
            acc += lengths[ep]
        else:
            train_eps.append(ep)

    def collect_indices(eps):
        idx = []
        for ep in eps:
            idx.extend(range(epi["from"][ep], epi["to"][ep]))
        return idx

    return (
        torch.utils.data.Subset(dataset, collect_indices(train_eps)),
        torch.utils.data.Subset(dataset, collect_indices(val_eps)),
        train_eps,
        val_eps,
    )
