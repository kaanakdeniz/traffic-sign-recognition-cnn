import matplotlib.pyplot as plt


def visualize_history(history):

    if "history" in history:
        history = history.history

    plt.style.use('default')
    plt.style.use('seaborn')

    plt.figure(0)
    plt.plot(history['accuracy'], label='train accuracy')
    plt.plot(history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('train_accuracy')

    plt.figure(1)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('train_loss')

