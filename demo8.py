import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(diabetes.data.shape)