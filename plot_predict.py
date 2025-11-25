import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 1. Define model (matches training)
# --------------------------
class scPred(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        scPred_defaults = {
            'num_layers': 4,
            'input_dim': 5313,   # must match training
            'hidden_dim': 64,
            'output_dim': 1,
            'reg_lambda': 5e-4,
            'dropout_rate': 0.05,
            'learning_rate': 9e-5,
            'random_seed': 1024
        }
        scPred_defaults.update(kwargs)
        for key, value in scPred_defaults.items():
            setattr(self, key, value)

        torch.manual_seed(self.random_seed)

        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout_rate)]
        hidden_layer = [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout_rate)]
        for _ in range(self.num_layers - 1):
            layers.extend(hidden_layer)
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------
# 2. Load trained model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = scPred()
state_dict = torch.load("/home/aliya/Liver/1111/scPred_inthep_96.pt", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# --------------------------
# 3. Prepare data
# --------------------------
df = pd.read_csv("/home/aliya/Liver/1111/HG00096_inthep_train_final_ballerina.csv")

# Feature and label extraction (must match indices used in training)
feature_start, feature_end = 2, 5315
test_set = ["chr12", "chr20", "chr5"]
test_data = df[df["chromo"].isin(test_set)]

test_epi = torch.tensor(test_data.iloc[:, feature_start:feature_end].to_numpy(dtype="float32")).to(device)
test_exp = torch.tensor(test_data.iloc[:, -1].to_numpy(dtype="float32")).to(device)

# --------------------------
# 4. Plot predictions
# --------------------------
with torch.no_grad():
    predictions = model(test_epi).cpu().numpy().flatten()
    true_vals = test_exp.cpu().numpy().flatten()

plt.scatter(true_vals, predictions, s=5)
plt.xlabel("True Expression")
plt.ylabel("Predicted Expression")
plt.title("Predicted vs True Expression")
plt.savefig("prediction_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Saved plot as prediction_plot.png")
