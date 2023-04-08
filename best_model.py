import random
import torch
import torch_geometric
from EFGNN import EFGNN
from train import train
from MultiHopTransform import OneTwoHopCSR
import torch_geometric.transforms as T

# please add CUBLAS_WORKSPACE_CONFIG=:4096:8 to environmental variables :)
# not that it helps though, for some reason I can't get it to be deterministic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(mode=True)
torch.manual_seed(0)
random.seed(0)

# list of (almost) all possible filters (by the old system)
all_filters = \
    [{"I": 1},  # 1
     {"I": 1, "h1": (1, -1, 0)}, {"I": 1, "h1": (1, -0.5, -0.5)}, {"I": 1, "h1": (1, 0, -1)},  # 4
     {"I": 1, "h1": (-1, -1, 0)}, {"I": 1, "h1": (-1, -0.5, -0.5)}, {"I": 1, "h1": (-1, 0, -1)},  # 7
     {"h1": (1, -1, 0)}, {"h1": (1, -0.5, -0.5)}, {"h1": (1, 0, -1)}, {"h1": (1, 0.5, 0)}, {"h1": (1, 1, 0)},  # 12
     {"I": 1, "h2": (1, -1, 0)}, {"I": 1, "h2": (1, -0.5, -0.5)}, {"I": 1, "h2": (1, 0, -1)},  # 15
     {"I": 1, "h2": (-1, -1, 0)}, {"I": 1, "h2": (-1, -0.5, -0.5)}, {"I": 1, "h2": (-1, 0, -1)},  # 18
     {"h2": (1, -1, 0)}, {"h2": (1, -0.5, -0.5)}, {"h2": (1, 0, -1)}, {"h2": (1, 0.5, -1.5)}, {"h2": (1, 1, -2)},  # 23
     {"h1": (1, -1.5, 0)}, {"h1": (1, -2, 0)},  # 25
     {"h1": (1, -0.5, -1)}, {"h1": (1, -0.5, -0.5)}, {"h1": (1, -0.5, 0)}, {"h1": (1, -0.5, 0.5)}, {"h1": (1, -0.5, 1)},
     # 30
     {"h1": (1, -1, -2)}, {"h1": (1, -1, -1)}, {"h1": (1, -1, 0)}, {"h1": (1, -1, 1)}, {"h1": (1, -1, 2)},  # 35
     {"h2": (1, -2, 0)}, {"h2": (1, -1.5, 0)},  # 37
     {"h2!": (1, -2, 0)}, {"h2!": (1, -1.5, 0)}, {"h2!": (1, -1, 0)}, {"h2!": (1, -0.5, -0.5)}, {"h2!": (1, 0, -1)},
     {"h2!": (1, 0.5, -1.5)}, {"h2!": (1, 1, -2)}]


# select the right filters

def best_model_for(dataset_name: str = "Squirrel"):
    """
    Demonstrates the performance of EFGNN on the 'Squirrel' and 'Chameleon' datasets.
    Args:
        dataset_name:
    """
    # Prepare the dataset

    assert dataset_name in ["Squirrel", "Chameleon"], "Only 'Squirrel' and 'Chameleon' options are currently available."
    dataset = torch_geometric.datasets.WikipediaNetwork(root='./datasets/', name=dataset_name)
    data = dataset[0]
    mask_num = 0
    data.train_mask = data.train_mask[:, mask_num]
    data.val_mask = data.val_mask[:, mask_num]
    data.test_mask = data.test_mask[:, mask_num]

    data = OneTwoHopCSR()(data)
    data = T.ToDevice("cuda" if torch.cuda.is_available() else "cpu")(data)

    avg = 0
    n_tests = 5

    # Select filters
    use_filters = []
    if dataset_name == "Squirrel":
        use_filters = [0, 24, 7, 9, 37, 38]
    else:
        use_filters = [0, 24, 7, 9, 38, 40]

    filters = [all_filters[i] for i in use_filters]
    for seed in range(n_tests):
        # unfortunatelly this is not enough for determinism  ¯\_(ツ)_/¯ (I think cuda is another source of randomness??)
        torch.manual_seed(seed)
        random.seed(seed)

        model = EFGNN(data.x.shape[1], hid_dim=64, output_dim=dataset.num_classes, filters=filters, prop2_filts=[],
                      use_decoder=False, dp1=0.7, dp2=0.7, use_alpha=True, use_deg=False, noise_mult=0.03).to(device)
        torch.cuda.empty_cache()
        lrs = {
            0: (0.003, 0.001),
            200: (0.001, 0),
            500: (0.0001, 0),
        }

        acc_val, acc_test, _ = train(model, data, num_epochs=900, use_edge_index=True, variable_lrs=lrs,
                                     print_reports=False)
        acc_val = f"{acc_val:.5f}"
        acc_test = f"{acc_test:.5f}"
        avg += float(acc_test) / float(n_tests)
        pretty_alphas = [float(f"{i:.3f}") for i in model.alpha.tolist()]
        print(f"seed: {seed}, acc_val: {acc_val}, acc_test: {acc_test}, alpha: {pretty_alphas}")
    print(f"Average test accuracy: {avg}")


best_model_for('Chameleon')
