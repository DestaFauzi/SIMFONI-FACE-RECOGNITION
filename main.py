import cv2

# Inisialisasi pengenal wajah (gunakan cascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mulai streaming dari kamera (gunakan angka 0 untuk kamera bawaan)
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Konversi frame ke abu-abu (grayscale) untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Tampilkan frame yang telah dimodifikasi
    cv2.imshow('Face Detection', frame)

    # Berhenti jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
