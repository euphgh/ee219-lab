#align(center, text(18pt)[
  *AI Computer System Lab2 Report*
])
#align(center)[
    Name: Guanghui Hu \
    Studne ID: 2024234318 \
    Teacher: Siting Liu \
  ]

#set par(justify: true)

= Q1.1: Plot histograms of the weights

  #image("q1_1.png")

= Q1.2: weights ranges for all layers

#figure(```txt
conv1: 3-sigma = [-0.31, 0.32], actual = [-1.21, 1.05]
conv2: 3-sigma = [-0.14, 0.07], actual = [-0.83, 0.75]
fc1: 3-sigma = [-0.01, 0.0], actual = [-0.25, 0.24]
fc2: 3-sigma = [-0.02, 0.02], actual = [-0.34, 0.36]
fc3: 3-sigma = [-0.15, 0.15], actual = [-0.59, 0.67]
```)

= Q2.2: Accuracy after quantizing weights
Accuracy of the network after quantizing all weights: 60.92%

= Q3.2: activations ranges for all layers

#figure(```txt
input: 3-sigma = [-0.81, 0.7], actual = [-1.0, 1.0]
conv1: 3-sigma = [-3.39, 5.04], actual = [0.0, 17.41]
conv2: 3-sigma = [-6.11, 7.41], actual = [0.0, 27.13]
fc1: 3-sigma = [-4.18, 4.96], actual = [0.0, 25.98]
fc2: 3-sigma = [-4.95, 6.36], actual = [0.0, 19.78]
fc3: 3-sigma = [-70.31, 70.18], actual = [-9.82, 15.15]
```)

= Q4.1: Equation for conv1 and conv2 layers' output

$ "Out"_"conv1" = frac("W1", S_"W1") times frac(\In, S_(\in)) times frac(S_"W1" times S_(\in) times 128, "max"_("conv1_activation")) $

$ "Out"_"conv2" = frac("W2", S_"W2") times frac("Out_conv1", W_("conv2")) times frac("max"_("conv1_activation"), S_"W1" times S_(\in) times 128) times frac(S_"W2" times S_"conv2" times 128, "max"_("conv2_activation")) $

= Q4.4: Accuracy with quantized weights and activations
Accuracy of the network after quantizing all weights: 60.95%

= Q5.1: Equation for the quantized layer with a bias

$ "bias"_"quan"  = frac("bias", S_"in" times S_"w1" times ... times S_"w5" times S_"out1" times ... times S_"out2") $

= Q5.1: Accuracy with quantized weights but unquantized bias
Accuracy of the network after quantizing all weights: 60.44%

= Q5.3: Accuracy with quantized weights and bias
Accuracy of the network after quantizing all weights: 60.8%

