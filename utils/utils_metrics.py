from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# def f1_score(y_true, y_pred):
#     """Compute the F1 score."""
#     return f1_score(y_true, y_pred)
#
#
# def accuracy_score(y_true, y_pred):
#     """Compute the F1 score."""
#     return accuracy_score(y_true, y_pred)
#
#
# def precision_score(y_true, y_pred):
#     """Compute the precision."""
#     return precision_score(y_true, y_pred)
#
#
# def recall_score(y_true, y_pred):
#     """Compute the recall."""
#     return recall_score(y_true, y_pred)


def find_best_threshold(all_predictions, all_labels):
    """find the best threshold between 0~1"""
    # all_predictions = np.ravel(all_predictions)
    # all_labels = np.ravel(all_labels)
    # get 100 alternative thresholds at 0.01 intervals from 0 to 1
    thresholds = [i / 100 for i in range(100)]
    all_f1s = []
    for threshold in thresholds:
        preds = (all_predictions >= threshold).astype("int")
        f1 = f1_score(y_true=all_labels, y_pred=preds)
        all_f1s.append(f1)
    # find the best threshold
    best_threshold = thresholds[int(np.argmax(np.array(all_f1s)))]
    print("best threshold is {}".format(str(best_threshold)))
    # print(all_f1s)
    return best_threshold


def classification_report(y_true, y_pred, digits=5):
    """Build a text report showing the main classification metrics."""
    # print("*"*50,"\n",true_entities,"\n","*"*50)
    # print("*"*50,"\n",pred_entities,"\n","*"*50)
    name_width = 0

    width = max(name_width, digits)

    headers = ["precision", "recall", "f1-score", "accuracy"]
    head_fmt =  u' {:>9}' * (len(headers)+1)
    report = '\n'
    report += head_fmt.format(u'', *headers, width=width)
    report += u'\n'

    row_fmt = u' {:>9.{digits}f}' * len(headers)

    report += u'\n'

    # compute
    report += row_fmt.format(precision_score(y_true, y_pred),
                             recall_score(y_true, y_pred),
                             f1_score(y_true, y_pred),
                             accuracy_score(y_true, y_pred),
                             width=width, digits=digits)

    return report
