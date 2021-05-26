import matplotlib.pyplot as plt

with open('results.txt', 'r') as f:
    lines = f.readlines()
    means = eval(lines[1])
    stds = eval(lines[3])

ticks = ['Vanilla', 'Vanilla optimized', 'Resnet', 'Siamese', 'Resnet auxiliary', 'Siamese auxiliary', 'Siamese semi auxiliary']
plt.bar(x=ticks, height=means.values())
for i, (k, v) in enumerate(stds.items()):
    plt.plot([ticks[i],ticks[i]], [means[k]+v, means[k]-v], c='k', linewidth=2)
plt.xticks(ticks,rotation=45)
plt.title('Model performances')
plt.ylabel('Average test accuracy')
plt.tight_layout()
plt.show()

b = {}
