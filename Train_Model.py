import torch.nn as nn
import torch, time
from torch.utils.data import Dataset, DataLoader
import h5py
from torch.autograd import Variable
from CNN_model import CNN
import numpy as np
import matplotlib.pyplot as plt

class H5Dataset(Dataset):

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        # print(index)
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


def load_data(batch_size):
    f = h5py.File("Sum_NewData_299_100.h5", 'r')
    X_train = f['Xtrain'][0:80000]
    Y_train = f['Ytrain'][0:80000]
    X_val = f['Xtrain'][80000:90000]
    Y_val = f['Ytrain'][80000:90000]
    train_set = H5Dataset(X_train, Y_train)
    val_set = H5Dataset(X_val, Y_val)
    train_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, pin_memory=True, shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=16, pin_memory=True, shuffle=False)
    return train_data_loader, val_data_loader
    

def test(model):
    f = h5py.File("Sum_NewData_299_100.h5", 'r')
    X_test = f['Xtrain'][90000:100000]
    Y_test = f['Ytrain'][90000:100000]
    test_set = H5Dataset(X_test, Y_test)
    test_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, pin_memory=True, shuffle=False)
    out_pre = np.zeros((10000, 299))
    ttime = 0
    for i, (data, label) in enumerate(test_data_loader):
        X_train, Y_train = Variable(data.unsqueeze(1)).cuda(), Variable(label).cuda()
        st = time.time()
        test_out = model(X_train)
        et = time.time()
        ttime += (et - st)
        out_pre[i:i+1] = test_out.detach().cpu().numpy()

    print('mean time:' + str(ttime*1000/10000) + 'ms')
    compute_rmse(out_pre, Y_test)


def compute_rmse(out_pre, test_label):
    num = np.shape(out_pre)[0]
    mse = np.zeros(num)
    for i in range(num):
        mse[i] = np.sum((out_pre[i] - test_label[i])**2)/299
    meanmse = np.sum(mse)/num
    meanrmse = np.sum(np.sqrt(mse))/num
    print('mse:', meanmse, '              rmse:', meanrmse)
    

def train(model, optimizer, scheduler, criterion, train_loader, val_loader, epoch, error, val_err, vla):

    for e in range(epoch):

        for i, (data, label) in enumerate(train_loader):
            X_train, Y_train = Variable(data.unsqueeze(1)).cuda(), Variable(label).cuda()
            model.train()
            optimizer.zero_grad()
            Y_pre = model(X_train)
            loss = criterion(Y_pre, Y_train)
            loss.backward()
            optimizer.step()
            error.append(loss.item())
            if i % 200 == 0:
                print('Epoch and Iterations::' + str(e) + ',' + str(i) + '   ' + 'MSE Loss:' + str(loss.item()))

        llr = 0
        val_loss = 0
        for i, (data, label) in enumerate(val_loader):
            X_train, Y_train = Variable(data.unsqueeze(1)).cuda(), Variable(label).cuda()
            out_val = model(X_train)
            val_error = criterion(out_val, Y_train)
            val_loss += val_error.item()
        val_loss = val_loss / len(val_loader)
        val_err.append(val_loss)
        if val_loss < vla:
            vla = val_loss
        scheduler.step(vla)
        print('Epoch:' + str(e) + '   ' + 'Val Loss:' + str(val_loss))

    return model, error, val_err


def plot_loss(train_err, val_err):
    x1 = np.arange(1, len(val_err)+1)
    x2 = np.arange(1, len(train_err)+1)
    x2 = x2 / 2500
    plt.plot(x2, train_err, 'b', x1, val_err, 'r')
    plt.xlabel('epochs')
    plt.ylabel('MSE LOSS')
    plt.title('MSE error of Zer')
    plt.show()


if __name__ == '__main__':
    train_err = []
    val_err = []
    lr = []
    vla = float('inf')
    LR = 1e-4
    epoch = 200
    batch_size = 32
    train_data_loader, val_data_loader = load_data(batch_size)
    model = CNN(1, 299).cuda()
    for name, param in model.named_parameters():
        param.data = param.data.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    criterion = nn.MSELoss()
    model, error, val_err = train(model, optimizer, scheduler, criterion, train_data_loader, val_data_loader, epoch, train_err, val_err, vla)
    model.eval().cpu()
    torch.save(model, 'CNN.pkl')
    x = torch.rand((1, 1, 240, 240), dtype=torch.float64)
    out = torch.onnx._export(model, x, "CNN.onnx", verbose=True)
    plot_loss(error, val_err)

    ############# test ####################
    # test(model)
