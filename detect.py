from ultralytics import YOLO
import cv2

# Load model YOLOv8 (versi kecil agar cepat)
model = YOLO("yolov8n.pt")

# Daftar kelas kendaraan yang ingin dihitung
vehicle_classes = ["car", "motorcycle", "bus", "truck", "bicycle"]

def detect_vehicle(image_path):
    # Jalankan inferensi YOLO
    results = model(image_path)[0]

    # Baca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Variabel menghitung jumlah kendaraan
    count = 0

    # Loop hasil deteksi
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]

        # Jika yang terdeteksi adalah kendaraan
        if label in vehicle_classes:
            count += 1

            # Ambil koordinat bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Gambar kotak di sekitar objek
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    # Simpan gambar hasil deteksi
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, img)

    # Return jumlah kendaraan + path gambar output
    return count, output_path
