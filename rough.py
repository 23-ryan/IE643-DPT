import matplotlib.pyplot as plt

plt.style.use('dark_background')

x = [2**i for i in range(1, 7)]
y = [28.4, 32.5, 33.0, 33.1, 45.2, 45.9]


plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='red')
plt.plot(x, y)
plt.xlabel('Number of samples per class')
plt.ylabel('Accuracy')
plt.title('Accuracy V/S Numshots [EPOCH=15]')
plt.show()