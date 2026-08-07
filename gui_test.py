# import cv2
# import numpy as np

# img = np.zeros((500, 500, 3), dtype=np.uint8)

# while True:
#     cv2.imshow("TEST WINDOW", img)

#     key = cv2.waitKey(30)
#     if key == ord('q'):
#         break

# cv2.destroyAllWindows()

import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())