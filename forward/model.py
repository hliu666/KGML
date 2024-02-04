import torch
import time
from tqdm import tqdm
# from .pytorchtools import EarlyStopping
# from .ANN import ANNModel
from pytorchtools import EarlyStopping
from ANN import ANNModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Model:
    def __init__(self, v, p):
        self.model = ANNModel(v.input_dim, p.hidden_dim, v.output_dim1, v.output_dim2)

    def train(self, dL, p):
        model = self.model
        train_loader, valid_loader = dL.train_loader, dL.valid_loader

        # Defining loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=p.learn_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=p.lr_decay)  # Define a learning rate scheduler

        epoch_times, tra_loss, val_loss = [], [], []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=p.patience, verbose=True)

        for epoch in tqdm(range(p.EPOCHS)):
            start_time = time.time()  # Start training loop

            train_loss = 0  # total loss of epoch
            train_count = 0

            model.to(device).train()  # Turn on the train mode
            for _, batch in enumerate(train_loader):
                X, target = batch

                X = X.float().to(device)
                target = target.float().to(device)

                predictions = model(X)

                loss = criterion(predictions, target)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()

                #for name, param in model.named_parameters():
                #    print(f"Gradient of {name}: {param.grad}")

                optimizer.step()

                train_loss += loss.item()
                train_count += 1

            # At the end of each epoch, decay learning rate
            if epoch % int(p.EPOCHS / 80) == 0:
                scheduler.step()

            model.to(device).eval()
            valid_loss = 0  # total loss of epoch
            valid_count = 0

            for _, batch in enumerate(valid_loader):
                X, target = batch

                X = X.float().to(device)
                target = target.float().to(device)

                predictions = model(X)  # Forward propagation

                loss = criterion(predictions, target)

                valid_loss += loss.item()  # Calculate softmax and cross entropy loss
                valid_count += 1

            current_time = time.time()
            epoch_times.append(current_time - start_time)

            tra_loss.append(train_loss / train_count)
            val_loss.append(valid_loss / valid_count)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(tra_loss, label="Training loss")
        ax.plot(val_loss, label="Validation loss")
        ax.legend()
        plt.show()

        return model

    def load(self, mdl, root=""):
        self.model.load_state_dict(torch.load(root + f'out/{mdl}_model.pth', map_location=torch.device('cpu')))
