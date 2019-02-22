import cv2
from Segmentation.Seg_Model import SegModel

def get_model(input_size=(320, 320, 3)):
    SegM = SegModel(input_size)
    Smodel = SegM.model
    Smodel.load_weights('Seg_weight.hdf5')
    l = len(Smodel.layers)
    for layer in Smodel.layers[:l]:
        layer.trainable = False

    return Smodel

img = cv2.imread('./hand.jpg')
orig_size = img.shape
res = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
res = np.expand_dims(res, axis=0)
model = get_model()
y_pred = model.predict(res)
cv2.imwrite("./hand_out.jpg", y_pred)
