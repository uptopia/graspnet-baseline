from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# # TensorBoard Visualizers
# TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
# TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))

writer = SummaryWriter('./runs')
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)

writer.add_text('text analysis', '今天天氣真好')