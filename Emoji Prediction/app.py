import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras

from data_preprocessing import *
from prediction import *


# Plot original label distribution before oversampling
plt.figure(figsize=(10, 6))
sns.countplot(x=Y_train)
plt.title('Original Label Distribution Before Oversampling')
plt.show()

# Plot label distribution after oversampling
plt.figure(figsize=(10, 6))
sns.countplot(x=Y_train_resampled)
plt.title('Label Distribution After Oversampling')
plt.show()
