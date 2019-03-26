import numpy as np
import matplotlib.pyplot as plt
from  statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample


np.random.seed(12345)
arparams = np.array([0.6])
maparams = np.array([0.3])
ar = np.r_[0.6, -arparams] # add zero-lag and negate



y = arma_generate_sample(ar, 500)

plt.plot(y)
plt.show()


model = ARIMA(y, (1, 0, 1)).fit(trend='nc', disp=0)
plt.plot(model.fittedvalues)
plt.show()
print('NP')
# # # fit model
# model = ARIMA(series, order=(5,1,0))
# model.
# model_fit = model.fit(disp=0)
# #
#