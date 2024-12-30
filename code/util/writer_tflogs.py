from torch.utils.tensorboard import SummaryWriter

def writerlog():
    writer = SummaryWriter(log_dir='../../tf-logs')
    return writer


# def writerlog():
#     writer = SummaryWriter(log_dir='../../autodl-tmp/tf-logs')
#     return writer


# def writerlog():
#     writer = SummaryWriter(log_dir='../../autodl-tmp/tf-logs_finetune')
#     return writer

# def writerlog():
#     writer = SummaryWriter(log_dir='../../tf-logs')
#     return writer
