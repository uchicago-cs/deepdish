import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size=6)
from pylab import *

import numpy as np
style.use('bmh')

t = np.linspace(-4, 4, 200)

figure(figsize=(7, 2))
xlimits = (-4, 4)
ylimits = (-.1, 1.1)
subplot(121)
title('Sigmoid')
plot(t, np.tanh(t)/2+0.5)
#plot(t, 1 - np.tanh(t)**2)
t1 = 0.4
t2 = 2
t3 = 4
ytext = 0.9
sett = dict(facecolor='0.5', alpha=0.1)
axvspan(-t1, t1, **sett)
axvspan(t2, t3, **sett)
axvspan(-t3, -t2, **sett)
xlim(xlimits)
ylim(ylimits)
sett = dict(fontsize=8, va='center', ha='center')
text(0, ytext, 'linear', **sett)
text(-(t2 + t3)/2., ytext, 'saturated', **sett)
text((t2 + t3)/2., ytext, 'saturated', **sett)

subplot(122)
title('ReLU')
plot(t, np.maximum(t, 0))
t1 = 0.4
t2 = 2.5
t3 = 4
ytext = 1.1
xlim(xlimits)
ylim(ylimits)

savefig('../public/images/activation-functions.svg')
