import matplotlib.pyplot as plt
def plot_results(results):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax[0].plot(results['train_loss_list'], 'go', linestyle='None')
    ax[0].plot(results['valid_loss_list'], 'bo', linestyle='None')
    ax[0].legend(['train loss', 'valid loss'])
    ax[0].set_title('Losses')
    ax[0].set_xlabel('Epoch')

    ax[1].plot(results['train_accuracy_list'], 'go', linestyle='None')
    ax[1].plot(results['valid_accuracy_list'], 'bo', linestyle='None')
    ax[1].legend(['train accuracy', 'valid accuracy'])
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel('Epoch')

    plt.show()