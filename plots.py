import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def plot_metric_confusion_matrix(y_test, logs, lim=30, figsize=(12, 5)):
    plt.figure(figsize=figsize)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    x_values = list(range(1, 300))[:lim]
    v1, v2, v3, v4 = list(zip(*list(map(lambda x: x['metrics'], logs))[:lim]))
    colors1 = ['coral', 'mintcream', 'mistyrose', 'lightblue']
    colors = ['darksalmon', 'lightcyan', 'midnightblue', 'slateblue']

    # Plotting
    plt.xlabel('k')
    plt.ylabel('Score')
    mean = np.vstack([v1, v2, v3, v4]).transpose().mean(axis=1)
    optimum = np.argmax(mean) + 1
    ax1 = plt.twinx()
    ax1.axvline(optimum, label='optimum', color=colors1[0], linestyle=':')
    ax1.set_yticks([])
    plt.plot(x_values, v1, label='Accuracy', color=colors1[0], linewidth=2, zorder=10)
    plt.plot(x_values, v2, label='F1 Score', linestyle='-', color=colors[3], linewidth=2)
    plt.plot(x_values, v3, label='Recall', linestyle=':', color=colors[0], linewidth=2)
    plt.plot(x_values, v4, label='Precision', linestyle=':', color=colors1[3], linewidth=2)
    plt.plot(x_values, mean, label='mean', linewidth=3)
    plt.scatter(optimum, mean[optimum - 1], marker='x', linewidth=4, color='tab:red', zorder=20)
    # Adding labels, title, and legend
    plt.xticks(list(range(1, lim + 1, 3)))
    plt.title('Performance Metrics')
    plt.legend()
    plt.subplot(1, 2, 2)

    logs = logs[optimum]['logs']
    predictions = logs['predictions']

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Plotting the confusion matrix with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    # Add axis labels, title, and tick marks for the confusion matrix plot
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], labels=['Class 0', 'Class 1'])
    plt.yticks([0.5, 1.5], labels=['Class 0', 'Class 1'], rotation=0)

    plt.tight_layout()
    plt.show()
    return fig


def plot_knn(predict_func, k, x_train, y_train, x_test, y_test, show_plot=False):
    logs = predict_func(k=k, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    metrics_class_0 = logs['metrics']['class0']
    metrics_class_1 = logs['metrics']['class1']
    predictions = logs['predictions']
    logs = dict(metrics=[metrics_class_0[0], *np.vstack([metrics_class_1[1:], metrics_class_0[1:]]).mean(axis=0)],
                predictions=predictions, logs=logs, fig=None)
    if show_plot:
        logs['fig'] = plot_metric_confusion_matrix(y_test=y_test, logs=logs, )

    return logs


def plot_pca(X, pca_func, y):
    np.random.seed(24)
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    reduced = pca_func(X - np.mean(X, axis=0))

    components_numpy = reduced['eigenvectors']
    components_numpy = np.array([components_numpy[:, 0], -components_numpy[:, 1]])
    components_numpy = pca_func(components_numpy, 2)
    components_numpy = components_numpy['matrix'].transpose()

    components_scikit = pca.components_
    components_scikit = pca_func(components_scikit, 2)
    components_scikit = components_scikit['matrix'].transpose()

    eigenvalues = np.abs(reduced['eigenvalues'])
    explained_variance = eigenvalues / eigenvalues.sum()

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    plt.title('PCA Reduced Dataset')
    x = reduced['matrix']
    plt.scatter(x[:, 0], -x[:, 1], label='numpy', c=y, cmap='viridis')
    plt.arrow(0, 0, -components_numpy[:, 0][0], -components_numpy[:, 1][0], head_width=.05, color='tab:blue', zorder=20)
    plt.arrow(0, 0, -components_numpy[:, 0][1], -components_numpy[:, 1][1], head_width=.05, color='tab:blue', zorder=20,
              label='numpy')
    plt.arrow(0, 0, components_scikit[:, 0][0], components_scikit[:, 1][0], head_width=.05, color='tab:orange',
              zorder=10, linewidth=2)
    plt.arrow(0, 0, components_scikit[:, 0][1], components_scikit[:, 1][1], head_width=.05, color='tab:orange',
              zorder=10, label='scikit', linewidth=2)
    plt.text(components_scikit[0, 1] / 2 + 0.1, components_scikit[1, 1] + 0.2, 'PCA2', fontsize=12, rotation=28,
             color='black', ha='center')
    plt.text(components_scikit[0, 0] / 2, components_scikit[1, 0] + 0.1, 'PCA2', fontsize=12, rotation=-28,
             color='black', ha='center')
    plt.axhline(0, zorder=-1, linewidth=.1, linestyle=':', color='black')
    plt.axvline(0, zorder=-1, linewidth=.1, linestyle=':', color='black')
    plt.legend()
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.title('Explained Variance')
    bars = plt.barh(['PCA1', 'PCA2'], explained_variance)
    for bar in bars:
        width = bar.get_width()
        percentage = f'{width:.2%}'
        plt.annotate(percentage,
                     xy=(width / 2, bar.get_y() + bar.get_height() / 2),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='center', color='white')
