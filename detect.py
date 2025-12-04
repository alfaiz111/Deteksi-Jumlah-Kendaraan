from ultralytics import YOLO
import cv2

# Load model YOLOv8 (Large untuk akurasi maksimal)
model = YOLO("yolov8l.pt")  # l = large

# Daftar kelas kendaraan
vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle", "scooter", "van", "train"]

def detect_vehicle(image_path):
    # Jalankan inferensi dengan confidence threshold 0.3 dan resize gambar lebih besar untuk objek kecil
    results = model(image_path, conf=0.3, imgsz=1280)[0]  # imgsz lebih besar untuk deteksi lebih akurat

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
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Gambar kotak di sekitar objek
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # Tambahkan jumlah kendaraan di atas gambar
    cv2.putText(img, f"Total Vehicles: {count}", 
                (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2)

    # Simpan gambar hasil deteksi
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, img)

    return count, output_path

# Contoh pemanggilan fungsi
if __name__ == "__main__":
    image_path = "test_image.jpg"  # ganti dengan path gambar kamu
    total, output_img = detect_vehicle(image_path)
    print(f"Jumlah kendaraan terdeteksi: {total}")
    print(f"Gambar hasil deteksi tersimpan di: {output_img}")
