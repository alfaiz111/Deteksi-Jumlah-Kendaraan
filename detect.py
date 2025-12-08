from ultralytics import YOLO
import cv2

# Load model YOLOv8 Large
model = YOLO("yolov8l.pt")

# Daftar kelas kendaraan yang ingin dihitung (sesuai COCO)
vehicle_classes = ["car", "truck", "bus", "motorcycle"]

def detect_vehicle(image_path):
    # Confidence agak rendah supaya kendaraan buram tetap terdeteksi
    results = model(image_path, conf=0.25, imgsz=1280)[0]

    img = cv2.imread(image_path)

    total_vehicle = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]     # Nama kelas asli YOLO

        # Hitung hanya kendaraan
        if label in vehicle_classes:
            total_vehicle += 1

        # Ambil koordinat bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Gambar kotak
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label TANPA angka (hanya nama kendaraan)
        cv2.putText(img, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    # Tampilkan total kendaraan di atas gambar
    cv2.putText(img, f"Total Vehicles: {total_vehicle}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    # Simpan output
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, img)

    return total_vehicle, output_path


# Contoh pemanggilan fungsi
if __name__ == "__main__":
    total, output_img = detect_vehicle("test_image.jpg")
    print("Jumlah kendaraan:", total)
    print("Output:", output_img)
