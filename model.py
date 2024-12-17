import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from get_XYZ import *
from utils import *
from tqdm import tqdm
import sys


class MagCompDataset(Dataset):
    def __init__(self, xyz, ind, features):
        arrays = []
        for key, value in xyz.items():
            if key in features:
                arrays.append(min_max_normalize(value[ind]))
        self.x = torch.tensor(np.vstack(arrays).T, dtype=torch.float32)
        self.y = torch.tensor(min_max_normalize(xyz['mag_1_c'][ind]), dtype=torch.float32)
        self.length = self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class Model1(nn.Module):
    def __init__(self, input_dim):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class Model2d(nn.Module):
    def __init__(self, input_dim):
        super(Model2d, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        b_total = x[:, -3:].sum(dim=1).unsqueeze(1)
        b_total = self._bpf(b_total)
    
        A = self._cal_matrix_A(x[:, -3:])
        A = self._bpf(A)

        beta_tl = b_total / A

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        output = (x + beta_tl) * A
        return output.sum(dim=1)
    
    def _bpf(self, x):
        return x
    
    def _cal_matrix_A(self, B):
        B_magnitude = torch.norm(B, dim=1, keepdim=True)
        B_hat = B / B_magnitude
        Bx_hat, By_hat, Bz_hat = B_hat[:, 0], B_hat[:, 1], B_hat[:, 2]
        sym_terms = torch.stack([
            B_magnitude.squeeze() * Bx_hat * Bx_hat,
            B_magnitude.squeeze() * Bx_hat * By_hat,
            B_magnitude.squeeze() * Bx_hat * Bz_hat,
            B_magnitude.squeeze() * By_hat * By_hat,
            B_magnitude.squeeze() * By_hat * Bz_hat,
            B_magnitude.squeeze() * Bz_hat * Bz_hat,
        ], dim=1)
        full_terms = torch.stack([
            B_magnitude.squeeze() * Bx_hat * Bx_hat,
            B_magnitude.squeeze() * Bx_hat * By_hat,
            B_magnitude.squeeze() * Bx_hat * Bz_hat,
            B_magnitude.squeeze() * By_hat * Bx_hat,
            B_magnitude.squeeze() * By_hat * By_hat,
            B_magnitude.squeeze() * By_hat * Bz_hat,
            B_magnitude.squeeze() * Bz_hat * Bx_hat,
            B_magnitude.squeeze() * Bz_hat * By_hat,
            B_magnitude.squeeze() * Bz_hat * Bz_hat,
        ], dim=1)
        delta = torch.cat([B_hat, sym_terms, full_terms], dim=1)
        return delta


if __name__ == '__main__':
    flight = "Flt1006"
    df_flight_path = "datasets/dataframes/df_flight.csv"
    df_flight = pd.read_csv(df_flight_path)
    xyz = get_XYZ(flight, df_flight)
    print(xyz.keys())
    df_all_path = "datasets/dataframes/df_all.csv"
    df_all = pd.read_csv(df_all_path)
    df_options = df_all[df_all['flight'] == flight]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    line = 1006.08  # test line
    batch_size = 32
    lr = 0.01
    epochs = 2

    test_ind = get_ind(xyz, tt_lim=[df_options[df_options['line'] == line]['t_start'].values[0],
                                    df_options[df_options['line'] == line]['t_end'].values[0]])  # get Boolean indices
    train_ind = ~test_ind
    exclude_features = ['line', 'flight', 'year', 'doy', 'tt', 'drape', 'mag_1_c', 'mag_1_lag', 'mag_1_dc', 'mag_1_igrf',
                        'mag_1_uc', 'ogs_mag', 'ogs_alt', 'radar', 'topo', 'flux_b_x', 'flux_b_y', 'flux_b_z']
    fields20 = "./datasets/fields_sgl_2020.csv"
    features = pd.read_csv(fields20, header=None).squeeze("columns").astype(str).tolist()
    features = [item for item in features if item not in exclude_features] + ['flux_b_x', 'flux_b_y', 'flux_b_z']
    print(features)
    dataset_train = MagCompDataset(xyz, train_ind, features)
    print("train num:", dataset_train.__len__())
    dataset_val = MagCompDataset(xyz, test_ind, features)
    print("val num:", dataset_val.__len__())
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = dataset_train.x.shape[1]
    model_type = 'Model1'
    print("input dim:", input_dim)
    # model = Model1(dataset_train.x.shape[1]).to(device)
    model = Model2d(input_dim).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    print("=" * 50)
    print("Start training...\n")
    sum_loss = 0.0
    model.train()
    for epoch in range(epochs):
        with tqdm(total=len(dataloader_train), desc='epoch{} [train]'.format(epoch + 1), file=sys.stdout) as t:
            for i, data in enumerate(dataloader_train):
                x, y = data
                x, y = x.to(device), y.to(device)
                y_hat = model(x).reshape(-1)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                t.set_postfix(loss=sum_loss / (i + 1), lr=scheduler.get_last_lr()[0])
                t.update(1)
            scheduler.step()

    print("=" * 50)
    print("\nStart validation...\n")
    model.eval()
    mag_error = 0.0
    with torch.no_grad():
        for data in dataloader_val:
            x, y = data
            x, y = x.to(device), y.to(device)
            y_hat = model(x).reshape(-1)
            mag_error += torch.sum(abs(y_hat - y))

    print("{}'s MagError:{}".format(model_type, mag_error / dataset_val.__len__()))
