
import matplotlib.pyplot as plt

def plot_2d_slice(img):
    plt.imshow(img, cmap='gray', origin='lower')
    plt.xlabel('First axis')
    plt.ylabel('Second axis')
    plt.colorbar(label='Signal intensity')
    plt.show()
