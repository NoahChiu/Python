from sklearn import datasets, model_selection
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
data = iris.data
target = iris.target

reg = LogisticRegression()
score = model_selection.cross_val_score(reg, data, target ,cv=3)# cv=3 將資料分成3等分交叉驗證
print(score)
