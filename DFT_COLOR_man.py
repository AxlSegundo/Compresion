import numpy as np
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
# Relleno de ceros para convertir a potencias de 2
def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def pad_to_power_of_2(image):
    rows, cols = image.shape
    new_rows = next_power_of_2(rows)
    new_cols = next_power_of_2(cols)
    padded_image = np.zeros((new_rows, new_cols))
    padded_image[:rows, :cols] = image
    return padded_image

def fft_manual(x):
    N = x.shape[0]
    if N <= 1:
        return x
    even = fft_manual(x[0::2])
    odd = fft_manual(x[1::2])
    T = np.exp(-2j * np.pi * np.arange(N) / N) * np.concatenate([odd, odd])
    return np.concatenate([even + T[:N // 2], even + T[N // 2:]])

def fft2_manual(image):
    # FFT de cada fila
    padded_image = pad_to_power_of_2(image)
    rows_fft = np.array([fft_manual(row) for row in tqdm(padded_image, desc="FFT Rows")])
    # FFT de cada columna
    cols_fft = np.array([fft_manual(col) for col in tqdm(rows_fft.T, desc="FFT Columns")]).T
    return cols_fft[:image.shape[0], :image.shape[1]]

def ifft_manual(x):
    x_conj = np.conjugate(x)
    result = fft_manual(x_conj)
    return np.conjugate(result) / x.shape[0]

def ifft2_manual(spectrum):
    padded_spectrum = pad_to_power_of_2(spectrum)
    # IFFT de cada fila
    rows_ifft = np.array([ifft_manual(row) for row in tqdm(padded_spectrum, desc="IFFT Rows")])
    # IFFT de cada columna
    cols_ifft = np.array([ifft_manual(col) for col in tqdm(rows_ifft.T, desc="IFFT Columns")]).T
    return cols_ifft[:spectrum.shape[0], :spectrum.shape[1]]

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
plt.imsave('IMG/lib/image_50.jpg', image_50)
plt.imsave('IMG/lib/image_80.jpg', image_80)
plt.imsave('IMG/lib/image_95.jpg', image_95)
