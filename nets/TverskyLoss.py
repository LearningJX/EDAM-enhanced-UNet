def tversky_loss(inputs, targets, beta=0.7):
    batch_size = targets.size(0)
    loss = 0.0

    for i in range(batch_size):
        prob = inputs[i]
        ref = targets[i]

        alpha = 1.0 - beta

        tp = (ref * prob).sum()
        fp = ((1 - ref) * prob).sum()
        fn = (ref * (1 - prob)).sum()
        tversky = tp / (tp + alpha * fp + beta * fn)
        loss = loss + (1 - tversky)
    return loss / batch_size
