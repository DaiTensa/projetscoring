fig , ax = plt.subplots(figsize=(10,10))
ax1 = plt.subplot2grid(shape=(3,3), loc=(0,0), rowspan=3)
ax2 = plt.subplot2grid(shape=(3,3), loc=(0,1), rowspan=3)

plt.tight_layout()
plt.show()