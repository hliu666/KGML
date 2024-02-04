import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import numpy as np


def prediction(model, test_loader, label_scaler):
    model.eval()
    for i, batch in enumerate(test_loader):
        X, target = batch
        out = model(X.float())  # Forward propagation

        if i == 0:
            y_pred = out.detach().numpy()
            y_test = target.detach().numpy()

        else:
            y_pred = np.vstack((y_pred, out.detach().numpy()))
            y_test = np.vstack((y_test, target.detach().numpy()))

    y_pred_scaler = label_scaler.inverse_transform(y_pred)
    y_test_scaler = label_scaler.inverse_transform(y_test)

    return y_pred_scaler.astype(float), y_test_scaler.astype(float)


def plot_bicm(model, test_loader, label_scaler):
    y_pred_scaler, y_test_scaler = prediction(model, test_loader, label_scaler)

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])
    r2, _ = pearsonr(y_test_scaler[:, 1], y_pred_scaler[:, 1])

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(y_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs00.plot(y_test_scaler[:, 0], color="b", label="Actual")
    axs00.set_title('GPP prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(y_pred_scaler[:, 0], y_test_scaler[:, 0], 1)
    axs01.scatter(y_pred_scaler[:, 0], y_test_scaler[:, 0], color="b")
    # You can also plot the regression line on top of this
    axs01.plot(y_pred_scaler[:, 0], k1 * y_pred_scaler[:, 0] + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(y_pred_scaler[:, 0], y_pred_scaler[:, 0], color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted GPP')
    axs01.set_ylabel('Modeled GPP')
    axs01.legend()

    axs10 = plt.subplot(gs[2])
    axs10.plot(y_pred_scaler[:, 1], "-o", color="g", label="Predicted")
    axs10.plot(y_test_scaler[:, 1], color="b", label="Actual")
    axs10.set_title('LST prediction: {0}'.format(round(r2 ** 2, 3)))
    axs10.legend()

    axs11 = plt.subplot(gs[3])
    k2, b2 = np.polyfit(y_pred_scaler[:, 1], y_test_scaler[:, 1], 1)
    axs11.scatter(y_pred_scaler[:, 1], y_test_scaler[:, 1], color="b")
    # Plot the regression line on top of this
    axs11.plot(y_pred_scaler[:, 1], k2 * y_pred_scaler[:, 1] + b2, color="g", label=f'y={k2:.2f}x+{b2:.2f}')
    axs11.plot(y_pred_scaler[:, 1], y_pred_scaler[:, 1], color="r", linestyle='dashed')
    axs11.set_xlabel('Predicted LST')
    axs11.set_ylabel('Modeled LST')
    axs11.legend()

    # after plotting the data, find the maximum and minimum values across all data
    min_01 = np.min([y_pred_scaler[:, 0].min(), y_test_scaler[:, 0].min()])
    max_01 = np.max([y_pred_scaler[:, 0].max(), y_test_scaler[:, 0].max()])

    min_11 = np.min([y_pred_scaler[:, 1].min(), y_test_scaler[:, 1].min()])
    max_11 = np.max([y_pred_scaler[:, 1].max(), y_test_scaler[:, 1].max()])

    # set the same x and y limits for axs10 and axs11
    axs01.set_xlim(min_01, max_01)
    axs01.set_ylim(min_01, max_01)

    axs11.set_xlim(min_11, max_11)
    axs11.set_ylim(min_11, max_11)

    plt.tight_layout()
    plt.show()


def plot_rtmo(model, test_loader, label_scaler):
    y_pred_scaler, y_test_scaler = prediction(model, test_loader, label_scaler)

    r1, _ = pearsonr(y_test_scaler[:, 0], y_pred_scaler[:, 0])
    r2, _ = pearsonr(y_test_scaler[:, 1], y_pred_scaler[:, 1])
    r3, _ = pearsonr(y_test_scaler[:, 2], y_pred_scaler[:, 2])

    fig = plt.figure(figsize=(18, 4))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1])

    axs00 = plt.subplot(gs[0])
    axs00.plot(y_pred_scaler[:, 0], "-o", color="g", label="Predicted")
    axs00.plot(y_test_scaler[:, 0], color="b", label="Actual")
    axs00.set_title('fPAR prediction: {0}'.format(round(r1 ** 2, 3)))
    axs00.legend()

    axs01 = plt.subplot(gs[1])
    k1, b1 = np.polyfit(y_pred_scaler[:, 0], y_test_scaler[:, 0], 1)
    axs01.scatter(y_pred_scaler[:, 0], y_test_scaler[:, 0], color="b")
    # You can also plot the regression line on top of this
    axs01.plot(y_pred_scaler[:, 0], k1 * y_pred_scaler[:, 0] + b1, color="g", label=f'y={k1:.2f}x+{b1:.2f}')
    axs01.plot(y_pred_scaler[:, 0], y_pred_scaler[:, 0], color="r", linestyle='dashed')
    axs01.set_xlabel('Predicted fPAR')
    axs01.set_ylabel('Modeled fPAR')
    axs01.legend()

    axs10 = plt.subplot(gs[2])
    axs10.plot(y_pred_scaler[:, 1], "-o", color="g", label="Predicted")
    axs10.plot(y_test_scaler[:, 1], color="b", label="Actual")
    axs10.set_title('red reflectance: {0}'.format(round(r2 ** 2, 3)))
    axs10.legend()

    axs11 = plt.subplot(gs[3])
    k2, b2 = np.polyfit(y_pred_scaler[:, 1], y_test_scaler[:, 1], 1)
    axs11.scatter(y_pred_scaler[:, 1], y_test_scaler[:, 1], color="b")
    # Plot the regression line on top of this
    axs11.plot(y_pred_scaler[:, 1], k2 * y_pred_scaler[:, 1] + b2, color="g", label=f'y={k2:.2f}x+{b2:.2f}')
    axs11.plot(y_pred_scaler[:, 1], y_pred_scaler[:, 1], color="r", linestyle='dashed')
    axs11.set_xlabel('Predicted red reflectance')
    axs11.set_ylabel('Modeled red reflectance')
    axs11.legend()

    axs20 = plt.subplot(gs[4])
    axs20.plot(y_pred_scaler[:, 2], "-o", color="g", label="Predicted")
    axs20.plot(y_test_scaler[:, 2], color="b", label="Actual")
    axs20.set_title('nir reflectance: {0}'.format(round(r2 ** 2, 3)))
    axs20.legend()

    axs21 = plt.subplot(gs[5])
    k2, b2 = np.polyfit(y_pred_scaler[:, 2], y_test_scaler[:, 2], 1)
    axs21.scatter(y_pred_scaler[:, 2], y_test_scaler[:, 2], color="b")
    # Plot the regression line on top of this
    axs21.plot(y_pred_scaler[:, 2], k2 * y_pred_scaler[:, 2] + b2, color="g", label=f'y={k2:.2f}x+{b2:.2f}')
    axs21.plot(y_pred_scaler[:, 2], y_pred_scaler[:, 2], color="r", linestyle='dashed')
    axs21.set_xlabel('Predicted nir reflectance')
    axs21.set_ylabel('Modeled nir reflectance')
    axs21.legend()

    # after plotting the data, find the maximum and minimum values across all data
    min_01 = np.min([y_pred_scaler[:, 0].min(), y_test_scaler[:, 0].min()])
    max_01 = np.max([y_pred_scaler[:, 0].max(), y_test_scaler[:, 0].max()])

    min_11 = np.min([y_pred_scaler[:, 1].min(), y_test_scaler[:, 1].min()])
    max_11 = np.max([y_pred_scaler[:, 1].max(), y_test_scaler[:, 1].max()])

    min_21 = np.min([y_pred_scaler[:, 2].min(), y_test_scaler[:, 2].min()])
    max_21 = np.max([y_pred_scaler[:, 2].max(), y_test_scaler[:, 2].max()])

    # set the same x and y limits for axs10 and axs11
    axs01.set_xlim(min_01, max_01)
    axs01.set_ylim(min_01, max_01)

    axs11.set_xlim(min_11, max_11)
    axs11.set_ylim(min_11, max_11)

    axs21.set_xlim(min_21, max_21)
    axs21.set_ylim(min_21, max_21)

    plt.tight_layout()
    plt.show()
