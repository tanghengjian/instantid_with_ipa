import copy
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import sys
img1=sys.argv[1]
img2=sys.argv[2]

# local load
#path="models/modelscope/cv_retinafce_recognition"  
# modelscope load
path="bubbliiiing/cv_retinafce_recognition"
face_recognition = pipeline("face_recognition", model=path, model_revision='v1.0.3')
emb1 = face_recognition(dict(user=img1))[OutputKeys.IMG_EMBEDDING]
emb2 = face_recognition(dict(user=img2))[OutputKeys.IMG_EMBEDDING]
sim = np.dot(emb1[0], emb2[0])
print(f'Face cosine similarity={sim:.3f}, img1:{img1}  img2:{img2}')