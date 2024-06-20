import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Funciones de FFT e IFFT manuales
def fft2_manual(image):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

def ifft2_manual(spectrum):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))

# Cargar la imagen a color
image_path = 'IMG/ciudad.jpg'
image = io.imread(image_path).astype(np.float32) / 255.0  # Normalizar a [0, 1]

# Aplicar DFFT manual a cada canal de la imagen para obtener el espectro
spectrum_r = fft2_manual(image[:, :, 0])
spectrum_g = fft2_manual(image[:, :, 1])
spectrum_b = fft2_manual(image[:, :, 2])

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

# Función para aplicar recorte y reconstrucción de la imagen
def apply_crop_and_ifft(spectrum, percentage):
    cropped_spectrum = crop_spectrum(spectrum, percentage)
    return np.abs(ifft2_manual(cropped_spectrum))

# Aplicar recortes del 50%, 80% y 95% a cada canal
image_50_r = apply_crop_and_ifft(spectrum_r, 50)
image_50_g = apply_crop_and_ifft(spectrum_g, 50)
image_50_b = apply_crop_and_ifft(spectrum_b, 50)

image_80_r = apply_crop_and_ifft(spectrum_r, 80)
image_80_g = apply_crop_and_ifft(spectrum_g, 80)
image_80_b = apply_crop_and_ifft(spectrum_b, 80)

image_95_r = apply_crop_and_ifft(spectrum_r, 95)
image_95_g = apply_crop_and_ifft(spectrum_g, 95)
image_95_b = apply_crop_and_ifft(spectrum_b, 95)

# Reconstruir imágenes a color después de recortar los espectros
image_50 = np.stack((image_50_r, image_50_g, image_50_b), axis=-1)
image_80 = np.stack((image_80_r, image_80_g, image_80_b), axis=-1)
image_95 = np.stack((image_95_r, image_95_g, image_95_b), axis=-1)

# Normalizar las imágenes resultantes a [0, 1]
image_50 = np.clip(image_50, 0, 1)
image_80 = np.clip(image_80, 0, 1)
image_95 = np.clip(image_95, 0, 1)

# Guardar las imágenes resultantes
plt.imsave('IMG/lib/al/image_50.jpg', image_50)
plt.imsave('IMG/lib/al/image_80.jpg', image_80)
plt.imsave('IMG/lib/al/image_95.jpg', image_95)
