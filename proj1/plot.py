import matplotlib.pyplot as plt
import numpy as np
with open('results.txt', 'r') as f:
    lines = f.readlines()
    means = eval(lines[1])
    stds = eval(lines[3])


plt.bar(x=means.keys(), height=[100*x  for x in means.values()])
for k, v in stds.items():
    plt.plot([k,k], [(means[k]+v)*100, (means[k]-v)*100], c='k', linewidth=2)
    plt.text(k, 100*means[k]+2, f'{100*means[k]:.2f}')
plt.xticks(rotation=45)
plt.title('Model performances')
plt.ylabel('Average test accuracy (%)')
plt.yticks(np.linspace(0,100,11))
plt.tight_layout()
plt.show()

