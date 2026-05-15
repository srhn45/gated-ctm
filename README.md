gated ctms

mnist
baseline: 326,860 params / train acc 0.973 / test acc 0.977
nongated tanh: 325,708 params / train acc 0.953 / test acc 0.966
nongated sigmoid: 325,708 params / train acc 0.977 / test acc 0.971
glu/sigmoid gate: 328,013 params / train acc 0.961 / test acc 0.975
glu/tanh gate: 328,013 params / train acc 0.973 / test acc 0.973

mazes
baseline: 8,561,144 params / test step acc 0.919 / test maze acc 0.511
nongated tanh: 8,543,736 params / 