import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import face_recognition
import mediapipe as mp
import os
import pickle

# Глобальная база данных лиц
DATABASE_PATH = "face_database.pkl"

# Загрузка/сохранение базы данных
def load_database():
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, "rb") as file:
            return pickle.load(file)
    return {}

def save_database(database):
    with open(DATABASE_PATH, "wb") as file:
        pickle.dump(database, file)

# GUI для обучения
def training_gui():
    def add_face():
        fio = fio_entry.get().strip()
        if not fio:
            messagebox.showerror("Ошибка", "Введите ФИО")
            return

        filepath = filedialog.askopenfilename(
            title="Выберите модель",
            filetypes=[("Проект", "*.jpg *.jpeg *.png")]
        )
        if not filepath:
            return

        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            messagebox.showerror("Ошибка", "Лицо не обнаружено!")
            return

        database = load_database()
        database[fio] = encodings[0]
        save_database(database)
        messagebox.showinfo("Успех", "Лицо добавлено в базу")

    window = tk.Tk()
    window.title("Обучение модели")
    tk.Label(window, text="Введите ФИО:").pack(pady=5)
    fio_entry = tk.Entry(window, width=30)
    fio_entry.pack(pady=5)

    add_button = tk.Button(window, text="Добавить лицо", command=add_face)
    add_button.pack(pady=10)

    window.mainloop()

# GUI для распознавания с 3D
def recognition_gui_with_3d():
    def start_recognition():
        database = load_database()
        if not database:
            messagebox.showerror("Ошибка", "База данных пуста!")
            return

        known_faces = list(database.values())
        known_names = list(database.keys())

        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        video_capture = cv2.VideoCapture(0)
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                # Распознавание лиц из базы данных
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_faces, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        match_index = matches.index(True)
                        name = known_names[match_index]

                    if name == "Unknown":
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


                # Наложение 3D-модели лица
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_iris_connections_style()
                        )

                # Отображение видео
                cv2.imshow("3D face recognition process", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        video_capture.release()
        cv2.destroyAllWindows()

    window = tk.Tk()
    window.title("Распознавание лиц с 3D-моделью")
    start_button = tk.Button(window, text="Запустить распознавание", command=start_recognition)
    start_button.pack(pady=20)
    window.mainloop()

# Главное меню
def main_menu():
    window = tk.Tk()
    window.title("Распознавание лиц")
    tk.Label(window, text="Выберите режим:").pack(pady=10)

    train_button = tk.Button(window, text="Обучение модели", command=training_gui)
    train_button.pack(pady=10)

    recognize_button_3d = tk.Button(window, text="Распознавание с 3D-моделью", command=recognition_gui_with_3d)
    recognize_button_3d.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    main_menu()
