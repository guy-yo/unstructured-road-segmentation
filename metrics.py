def precision(preds, targets):
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    return tp / (tp + fp + 1e-6)

def recall(preds, targets):
    tp = ((preds == 1) & (targets == 1)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    return tp / (tp + fn + 1e-6)
