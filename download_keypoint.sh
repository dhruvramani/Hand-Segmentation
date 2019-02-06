
# ------------------------- BODY, FACE AND HAND MODELS -------------------------
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/"
HAND_FOLDER="./utils/pose_iter_102000.caffemodel"

# "------------------------- HAND MODELS -------------------------"
# Hand
HAND_MODEL=$OPENPOSE_URL"pose_iter_102000.caffemodel"
wget -c ${HAND_MODEL} -P ${HAND_FOLDER}