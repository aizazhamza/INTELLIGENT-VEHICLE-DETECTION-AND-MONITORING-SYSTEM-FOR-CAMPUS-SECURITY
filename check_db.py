# import campus_db

# conn = campus_db.connect_database()

# campus_db.insert_event(
#     conn,
#     "TEST123",
#     "Entry",
#     "test.jpg",
#     0.9,
#     1
# )
# # import campus_db  # ← add this
# # db_conn = campus_db.connect_database()
# # print("Connected:", db_conn.is_connected())  # must print True

from ultralytics import YOLO

model = YOLO(r"E:\anpr_project\best.pt")

metrics = model.val(
    data=r"E:\LPD Pakistani Vehicle\LPD Pakistani Vehicle\data.yaml",
    imgsz=640,
    split="test"  # or "val" if your dataset doesn't have a test split
)