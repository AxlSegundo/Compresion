import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Funciones de FFT e IFFT manuales
def fft2_manual(image):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

def ifft2_manual(spectrum):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))

# Cargar la imagen en escala de grises
image_path = 'IMG/ciudad.jpg'
image = io.imread(image_path)
grayscale_image = color.rgb2gray(image)

# Aplicar DFFT manual a la imagen para obtener el espectro
spectrum = fft2_manual(grayscale_image)

# Guardar la imagen original y su espectro
plt.imsave('IMG/original_image.jpg', grayscale_image, cmap='gray')
magnitude_spectrum = np.log(np.abs(spectrum) + 1)
plt.imsave('IMG/magnitude_spectrum.jpg', magnitude_spectrum, cmap='gray')

# Función para recortar el espectro
def crop_spectrum(spectrum, percentage):
    rows, cols = spectrum.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros_like(spectrum)
    
    # Calcular el tamaño del recorte
    r = int(crow * percentage / 100)
    c = int(ccol * percentage / 100)
    
    mask[crow-r:crow+r, ccol-c:ccol+c] = 1
    return spectrum * mask

# Aplicar recortes del 50%, 80% y 95%
spectrum_50 = crop_spectrum(spectrum, 50)
spectrum_80 = crop_spectrum(spectrum, 80)
spectrum_95 = crop_spectrum(spectrum, 95)

# Inversa de la DFFT manual (IDFFT)
image_50 = np.abs(ifft2_manual(spectrum_50))
image_80 = np.abs(ifft2_manual(spectrum_80))
image_95 = np.abs(ifft2_manual(spectrum_95))

# Guardar las imágenes resultantes
plt.imsave('IMG/image_50.jpg', image_50, cmap='gray')
plt.imsave('IMG/image_80.jpg', image_80, cmap='gray')
plt.imsave('IMG/image_95.jpg', image_95, cmap='gray')
