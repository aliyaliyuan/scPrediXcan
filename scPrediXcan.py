import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

# specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and prepare data
print("Loading and preparing data")
gen_data_p = 'HG00096_inthep_train_final_ballerina.csv'
gen_data = pd.read_csv(gen_data_p, dtype=str)

#Chromo as string
gen_data['chromo'] = gen_data['chromo'].astype(str).str.strip()

if not gen_data['chromo'].str.startswith("chr").all() and "TSS_enformer_input" in gen_data.columns:
    gen_data['chromo'] = gen_data['TSS_enformer_input'].str.split("_".str[0])

#Identify feature columns
feature_cols = [col for col in gen_data.columns if col.startswith("feature_")]

#Converting features and mean_expression to numeric just in case they are not read in as numeric
gen_data[feature_cols] = gen_data[feature_cols].apply(pd.to_numeric, errors="coerce")
gen_data["mean_expression"] = pd.to_numeric(gen_data["mean_expression"], errors = "coerce")


#Convert NAs to 0
gen_data[feature_cols] = gen_data[feature_cols].fillna(0)

#Debug check
print(gen_data.dtypes.head(10))

# Split sets by chromosome
train_set = ["chr1", "chr10", "chr13", "chr15", "chr16", "chr17", "chr18", "chr19", "chr2", "chr21", "chr22", "chr3", "chr4", "chr6", "chr8", "chr9", "chrX", "chrY"]
val_set = ["chr11", "chr14", "chr7"]
test_set = ["chr12", "chr20", "chr5"]

def data_prepare(gen_data, train_set, val_set, test_set, is_normalization=True):
    # --- FIX 1: ensure all feature columns are numeric before splitting ---
    feature_start, feature_end = 2, 5315  # adjust if necessary
    feature_cols = gen_data.columns[feature_start:feature_end]

    # Coerce non-numeric to NaN and fill
    gen_data[feature_cols] = gen_data[feature_cols].apply(pd.to_numeric, errors='coerce')
    gen_data[feature_cols] = gen_data[feature_cols].fillna(0)

    # --- FIX 2: verify numeric conversion ---
    if not all(pd.api.types.is_numeric_dtype(gen_data[c]) for c in feature_cols):
        bad_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(gen_data[c])]
        raise ValueError(f" Still non-numeric columns: {bad_cols[:10]} ...")

    # split data
    train_data = gen_data[gen_data['chromo'].isin(train_set)]
    val_data = gen_data[gen_data['chromo'].isin(val_set)]
    test_data = gen_data[gen_data['chromo'].isin(test_set)]

    # convert to torch tensors
    train_epi = torch.tensor(train_data.iloc[:, feature_start:feature_end].to_numpy(dtype='float32')).to(device)
    train_exp = torch.tensor(train_data.iloc[:, -1].to_numpy(dtype='float32')).to(device)

    val_epi = torch.tensor(val_data.iloc[:, feature_start:feature_end].to_numpy(dtype='float32')).to(device)
    val_exp = torch.tensor(val_data.iloc[:, -1].to_numpy(dtype='float32')).to(device)

    test_epi = torch.tensor(test_data.iloc[:, feature_start:feature_end].to_numpy(dtype='float32')).to(device)
    test_exp = torch.tensor(test_data.iloc[:, -1].to_numpy(dtype='float32')).to(device)

    if is_normalization:
        # normalize input data
        scaler = StandardScaler().fit(train_epi.cpu())
        train_epi = torch.tensor(scaler.transform(train_epi.cpu()), dtype=torch.float32).to(device)
        val_epi = torch.tensor(scaler.transform(val_epi.cpu()), dtype=torch.float32).to(device)
        test_epi = torch.tensor(scaler.transform(test_epi.cpu()), dtype=torch.float32).to(device)

    return train_epi, train_exp, val_epi, val_epi, val_exp, test_epi, test_exp

# put the data into a iterator which reads data in batches
def dataloader(features, labels, batch_size, is_train = True):
    dataset = data.TensorDataset(features, labels)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)


train_epi, train_exp, val_epi, val_epi, val_exp, test_epi, test_exp = data_prepare(gen_data, train_set, val_set, test_set)
train_data_iter = dataloader(train_epi, train_exp, batch_size=1000)

print("Length of train_epi:")
len(train_epi)

print("Model training and saving")

# define the scPred model and some helper functions

class scPred(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        scPred_defaults = {
            'num_layers' : 4,
            'input_dim' : 5313,
            'hidden_dim' : 64,
            'output_dim' : 1,
            'reg_lambda' : 5e-4,
            'dropout_rate' : 0.05,
            'learning_rate' : 9e-5,
            'random_seed' : 1024
        }

        scPred_defaults.update(kwargs)

        for key, value in scPred_defaults.items():
            setattr(self, key, value)


        torch.manual_seed(self.random_seed)


        # model main
        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout_rate)]
        hidden_layer = [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout_rate)]
        
        for _ in range(self.num_layers - 1):
            layers.extend(hidden_layer)
        
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.net = nn.Sequential(*layers)

    
    def custom_loss(self, y_true, y_pred):
        return F.mse_loss(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))

    def forward(self, x):
        return self.net(x)
    
    def compile(self):
        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = self.reg_lambda)


def plot_loss_curve(train_losses, val_losses, epochs):

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train_loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def scPred_training(model, train_data_iter, val_epi, val_exp, epochs=80, plot_loss=True, model_path = '/home/aliya/Liver/1111/scPred_inthep_96.pt'):
    model.compile()
    optimizer = model.optimizer

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for _ in range(epochs):
        
        model.eval()
        with torch.no_grad():
            val_loss = model.custom_loss(model(val_epi), val_exp).item()
        val_losses.append(val_loss)

        
        model.train()  

        for batch_x, batch_y in train_data_iter:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = model.custom_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        train_losses.append(model.custom_loss(model(train_epi), train_exp).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, model_path)
        

    if plot_loss:
        plot_loss_curve(train_losses, val_losses, epochs)

scPred_test = scPred().to(device)

saved_path = '/home/aliya/Liver/1111/scPred_inthep_96.pt'

scPred_training(scPred_test, train_data_iter, val_epi, val_exp, epochs=100, model_path=saved_path)


#Evaluate performance on test set
print("Evaluating on test set")

def plot_prediction(model, test_features, test_labels):
    model.eval()  # eval mode

    with torch.no_grad():
        predictions = model(test_features)

    # Move tensors to CPU and convert to numpy
    predictions = predictions.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(test_labels, predictions, s=5)
    plt.xlabel("True Expression")
    plt.ylabel("Predicted Expression")
    plt.title("scPrediXcan Prediction Performance")
    plt.tight_layout()
    plt.show()

