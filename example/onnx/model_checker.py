import redisai as rai
from ml2rt import load_model
import numpy as np

model = load_model("model.onnx")
device = 'cpu'

con = rai.Client(host='localhost', port=6379)
con.modelset("model", 'onnx', device, model)
dummydata = np.array([[5.1, 3.5, 1.4, 0.2]]).astype(dtype=np.float32)
#print(dummydata.shape)

# get key in dict model
#print(con.modelget("model").keys())
con.tensorset("float_input", dummydata, shape=(1, 4), dtype='float')
con.modelrun("model", inputs=["float_input"], outputs=['output_label', 'output_probability'])
outtensor = con.tensorget("output_label")
print(outtensor)