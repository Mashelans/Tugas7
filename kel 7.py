import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. BACA CITRA ASLI
# ==========================================

img_path = r'C:\Users\mashela sari\OneDrive - Universitas Negeri Malang\Documents\Programming Computer\nyc.JPEG'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        f"File {img_path} tidak ditemukan. Pastikan nama file benar."
    )

# Konversi citra ke float untuk perhitungan
original = img.astype(np.float32)


# ==========================================
# 2. MENAMBAHKAN GAUSSIAN NOISE
#    σ = 10
# ==========================================

sigma = 10

gaussian_noise = np.random.normal(
    0,
    sigma,
    original.shape
).astype(np.float32)

# Tambahkan noise ke citra asli
noisy_img = original + gaussian_noise

# Batasi nilai piksel agar tetap 0–255
noisy_img = np.clip(
    noisy_img,
    0,
    255
)


# ==========================================
# 3. FILTER SPASIAL 3 x 3
# ==========================================

kernel = np.ones(
    (3, 3),
    np.uint8
)


# ------------------------------------------
# A. FILTER MEDIAN
# ------------------------------------------

median_img = cv2.medianBlur(
    noisy_img.astype(np.uint8),
    3
).astype(np.float32)


# ------------------------------------------
# B. FILTER MAXIMUM
# ------------------------------------------

max_img = cv2.dilate(
    noisy_img.astype(np.uint8),
    kernel
).astype(np.float32)


# ------------------------------------------
# C. FILTER MINIMUM
# ------------------------------------------

min_img = cv2.erode(
    noisy_img.astype(np.uint8),
    kernel
).astype(np.float32)


# ------------------------------------------
# D. FILTER TITIK TENGAH / MIDPOINT
# ------------------------------------------

midpoint_img = (
    max_img + min_img
) / 2.0


# ------------------------------------------
# E. FILTER RERATA ARITMETIK
# ------------------------------------------

kernel_mean = np.ones(
    (3, 3),
    np.float32
) / 9.0

arithmetic_mean = cv2.filter2D(
    noisy_img,
    -1,
    kernel_mean
)


# ------------------------------------------
# F. FILTER RERATA Yp / GEOMETRIC MEAN
# ------------------------------------------

eps = 1e-5

# Padding untuk mendapatkan jendela 3 x 3
padded = np.pad(
    noisy_img,
    1,
    mode='edge'
)

rows, cols = noisy_img.shape

geometric_mean = np.zeros_like(
    noisy_img
)

for i in range(rows):
    for j in range(cols):

        # Ambil jendela 3 x 3
        window = padded[
            i:i+3,
            j:j+3
        ]

        # Geometric mean
        log_window = np.log(
            window + eps
        )

        geometric_mean[i, j] = np.exp(
            np.mean(log_window)
        )


# ==========================================
# 4. MENGHITUNG MSE
#    DIBANDINGKAN DENGAN CITRA ASLI
# ==========================================

def hitung_mse(citra_asli, citra_hasil):

    return np.mean(
        (citra_asli - citra_hasil) ** 2
    )


# Menyimpan seluruh hasil filter
hasil_filter = {
    "Median": median_img,
    "Maximum": max_img,
    "Minimum": min_img,
    "Titik Tengah": midpoint_img,
    "Rerata Aritmetik": arithmetic_mean,
    "Rerata Yp": geometric_mean
}


# ==========================================
# 5. HITUNG MSE MASING-MASING FILTER
# ==========================================

print("========================================")
print("       HASIL PERBANDINGAN MSE")
print("========================================")

mse_results = {}

for nama, hasil in hasil_filter.items():

    mse = hitung_mse(
        original,
        hasil
    )

    mse_results[nama] = mse

    print(
        f"{nama}: MSE = {mse:.4f}"
    )


# ==========================================
# 6. MENENTUKAN FILTER TERBAIK
# ==========================================

filter_terbaik = min(
    mse_results,
    key=mse_results.get
)

mse_terbaik = mse_results[
    filter_terbaik
]

print("\n========================================")
print("          FILTER TERBAIK")
print("========================================")

print(
    f"Filter terbaik = {filter_terbaik}"
)

print(
    f"MSE terkecil    = {mse_terbaik:.4f}"
)


# ==========================================
# 7. VISUALISASI HASIL
# ==========================================

plt.figure(
    figsize=(15, 8)
)


# ------------------------------------------
# CITRA ASLI
# ------------------------------------------

plt.subplot(2, 4, 1)

plt.imshow(
    original,
    cmap='gray',
    vmin=0,
    vmax=255
)

plt.title("Citra Asli")

plt.axis('off')


# ------------------------------------------
# CITRA DENGAN GAUSSIAN NOISE
# ------------------------------------------

plt.subplot(2, 4, 2)

plt.imshow(
    noisy_img,
    cmap='gray',
    vmin=0,
    vmax=255
)

plt.title(
    "Gaussian Noise\nσ = 10"
)

plt.axis('off')


# ------------------------------------------
# HASIL FILTER
# ------------------------------------------

idx = 3

for nama, hasil in hasil_filter.items():

    plt.subplot(2, 4, idx)

    plt.imshow(
        hasil,
        cmap='gray',
        vmin=0,
        vmax=255
    )

    plt.title(
        f"{nama}\nMSE = {mse_results[nama]:.2f}"
    )

    plt.axis('off')

    idx += 1


plt.tight_layout()

plt.show()