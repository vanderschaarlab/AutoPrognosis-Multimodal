from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
import numpy as np


def get_metric(preds, labels, task, metric):
  if task ==  'binary':
    if metric == 'Accuracy':
      return accuracy_score(labels, np.argmax(preds, axis=1))
    elif metric == 'AUROC':
      return roc_auc_score(labels, preds[:,1])
    elif metric == 'F1 Score':
      return f1_score(labels, np.argmax(preds, axis=1))
    elif metric == 'Matt. Corr.':
      return matthews_corrcoef(labels, np.argmax(preds, axis=1))
    else:
      print('Invalid metric')

  elif task == 'multi-class':
    if metric == 'Accuracy':
      return accuracy_score(labels, np.argmax(preds, axis=1))
    elif metric == 'Bal. Acc.':
      return balanced_accuracy_score(labels, np.argmax(preds, axis=1))
    elif metric == 'AUROC':
      return roc_auc_score(labels, preds[:,1], multi_class="ovr")
    elif metric == 'F1 Score':
      return f1_score(labels, np.argmax(preds, axis=1), average="macro")
    else:
      print('Invalid metric')

  else:
    print('Invalid task')