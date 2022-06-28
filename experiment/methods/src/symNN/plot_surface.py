import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from de_learn_network import log_activation, L1L2_m

model = load_model('../models/model_div_eq.h5', custom_objects={'log_activation':log_activation,
                                                                'L1L2_m':L1L2_m})

# defining surface and axes
x = np.outer(np.linspace(10, 1000, 100), np.ones(100))
y = x.copy().T
print(y)
print(x)
pred_z = model.predict([np.ravel(x), np.ravel(y)])
pred_z = pred_z.reshape((100, 100))
print(pred_z.shape)
z = x/y

# actual plot
plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')
ax.set_title('y = x1/x2 + 2x1 - 5x2 actual surface')
plt.show()

# predicted plot
plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, pred_z, cmap ='plasma')
ax.set_title('y = x1/x2 + 2x1 - 5x2 model predicted surface')
plt.show()

