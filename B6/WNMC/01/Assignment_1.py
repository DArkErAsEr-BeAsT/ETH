import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(4.5,3)})

list_1 = random.uniform((-18944827/10000),(18944827/10000))
list_2 = random.uniform((-18944827/10000),(18944827/10000))
list_3 = random.uniform((-18944827/10000),(18944827/10000))
list_4 = random.uniform((-18944827/10000),(18944827/10000))
list_5 = random.uniform((-18944827/10000),(18944827/10000))

n = 10000
a = 0
b = 10
data_uniform = uniform.rvs(size=n, loc = a, scale=b)

ax = sns.distplot(data_uniform,
                  bins=100,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Uniform ', ylabel='Frequency')