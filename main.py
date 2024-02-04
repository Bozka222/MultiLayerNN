import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

#  --------------------- VARIABLE DECLARATION ---------------------
OUTPUT_LAYER_NEURONS_NUMBER = 10  # Number of neurons in output layer
FIRST_LAYER_NEURONS_NUMBER = 30  # Number of neurons in first hidden layer
SECOND_LAYER_NEURONS_NUMBER = 30  # Number of neurons in second hidden layer

MAX_ITERATION = 2000  # Maximal number of iterations
EC_STOP = 0.1  # Required error value to stop learning cycle
ALPHA = 0.005  # Learning coefficient
M = 0.1  # Network moment (weights update coefficient after every cycle)
NOISE_LEVEL = 0.3  # Level of noise added to images

height, width, _ = cv2.imread('sample_pics/0.bmp').shape  # Determines size of sample picture
PIC_SIZE = height*width  # Size of one picture

x = np.zeros((OUTPUT_LAYER_NEURONS_NUMBER, PIC_SIZE))  # Matrix for all sample pictures (one row, one pic)
x_noise = np.zeros((OUTPUT_LAYER_NEURONS_NUMBER, PIC_SIZE))  # Same matrix for noisy pictures

#  --------------------- INPUT SAMPLE LOAD AND NOISE CREATION ---------------------

for i in range(0, OUTPUT_LAYER_NEURONS_NUMBER):
    img_load = cv2.imread(f'sample_pics/{i}.bmp', cv2.IMREAD_GRAYSCALE)  # load sample images
    norm_image = cv2.normalize(img_load, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    x[i] = norm_image.flatten()

    img_noise = img_load  # create new image for noise addition
    noise_size = int(NOISE_LEVEL * PIC_SIZE)  # determine size of the noise (percentage of image size)
    random_indices = np.random.choice(PIC_SIZE, noise_size)  # Selecting random indices for noise
    noise = np.random.choice([img_load.min(), img_load.max()], noise_size)  # Creates noise list (max and min values)
    img_noise.flat[random_indices] = noise  # Replace values at random location with noise values
    norm_image_noise = cv2.normalize(img_noise, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    x_noise[i] = norm_image_noise.flatten()

d = np.eye(OUTPUT_LAYER_NEURONS_NUMBER)*2-1  # Expected output values

#  --------------------- WEIGHTS INITIALIZATION ---------------------

w0 = np.ones((1, PIC_SIZE))  # 324 input neurons
# (324 * 30) w1 + one row of w0
w1 = (np.random.uniform(low=0, high=1, size=(PIC_SIZE+1, FIRST_LAYER_NEURONS_NUMBER))-0.5)/100
# (30 * 30) w2 + one row of w0
w2 = (np.random.uniform(low=0, high=1, size=(FIRST_LAYER_NEURONS_NUMBER + 1, SECOND_LAYER_NEURONS_NUMBER))-0.5)/100
# (30 * 10) w3 + one row of w0
w3 = (np.random.uniform(low=0, high=1, size=(SECOND_LAYER_NEURONS_NUMBER + 1, 10))-0.5)/100

w1_old = w1  # First weights setting
w2_old = w2
w3_old = w3

#  --------------------- LEARNING ALGORITHM ---------------------

i = 0
Ec = 1
v_Ec = []
one = np.array([[1]])  # Number 1 adding value
global y0, y1, y2, y3

while (Ec > EC_STOP) and (i <= MAX_ITERATION):
    Ec = 0

    for j in range(0, OUTPUT_LAYER_NEURONS_NUMBER):
        # Input
        y0 = np.multiply(w0, x[j, :])
        y0_1 = np.hstack((y0, one))

        # Hidden layers
        y1 = np.tanh(np.dot(y0_1, w1))  # Matrix multiplication
        y1_1 = np.hstack((y1, one))  # Need to add one value with number 1
        y2 = np.tanh(np.dot(y1_1, w2))
        y2_1 = np.hstack((y2, one))

        # Output layer
        y3 = np.tanh(np.dot(y2_1, w3))

        # Error count
        error3 = d[j, :] - y3  # Error of output layer
        error2 = np.dot(error3, w3[: - 1, :].transpose())  # Backpropagation of error of 2.hidden layer
        error1 = np.dot(error2, w2[: - 1, :].transpose())  # Backpropagation of error of 1.hidden layer

        # Temporarily save weights
        w1_temp = w1
        w2_temp = w2
        w3_temp = w3

        # Count new weights
        w1 = w1 + y0_1.transpose() * ALPHA * np.multiply(error1, (1 - (y1 ** 2))) + M * (w1 - w1_old)
        w2 = w2 + y1_1.transpose() * ALPHA * np.multiply(error2, (1 - (y2 ** 2))) + M * (w2 - w2_old)
        w3 = w3 + y2_1.transpose() * ALPHA * np.multiply(error3, (1 - (y3 ** 2))) + M * (w3 - w3_old)

        # Save old weight values
        w1_old = w1_temp
        w2_old = w2_temp
        w3_old = w3_temp

        # Count partial error
        Ec = Ec + (np.sum(error3 ** 2) / 2)

    v_Ec.append(Ec)  # Save partial error
    i += 1

#  --------------------- NETWORK ERROR GRAPH ---------------------

plt.plot(v_Ec)
plt.title("Network Error History", fontweight='bold', fontsize=20)
plt.xlabel("Iterations [Number]")
plt.ylabel("Ec (Network Error)")
plt.grid()
plt.savefig("Output_Figures/Error_History.pdf", dpi=300)
plt.show()

#  --------------------- INPUT NOISY SAMPLES GRAPH ---------------------

fig, axarr = plt.subplots(1, 10, sharey=True, figsize=(5, 2))
for i in range(0, OUTPUT_LAYER_NEURONS_NUMBER):
    axarr[i].imshow(x_noise[i].reshape(height, width))
    plt.setp([a.get_xticklabels() for a in axarr[:]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:]], visible=False)
    fig.suptitle('Noisy Input Samples', fontsize=15, fontweight='bold')
plt.savefig("Output_Figures/Input_Noisy_Samples.pdf", dpi=300)
plt.show()

#  --------------------- NETWORK OUTPUT FOR NOISY SAMPLES ---------------------

fig, axarr = plt.subplots(1, 10, sharey=True, figsize=(12, 12))
x_label = [i for i in range(0, 10)]
print(x_label)
for i in range(0, OUTPUT_LAYER_NEURONS_NUMBER):
    y0 = np.multiply(w0, x_noise[i, :])
    y0_1 = np.hstack((y0, one))
    y1 = np.tanh(np.dot(y0_1, w1))
    y1_1 = np.hstack((y1, one))
    y2 = np.tanh(np.dot(y1_1, w2))
    y2_1 = np.hstack((y2, one))
    y3 = np.tanh(np.dot(y2_1, w3))
    axarr[i].bar(x_label, sum(y3))
    axarr[i].set_xlabel(f"Neuron {i}", fontsize=15, rotation=45)
    axarr[0].set_ylabel("Output Neuron Value", fontsize=20)
    fig.suptitle('Output Layer Values', fontsize=40, fontweight='bold')
plt.savefig("Output_Figures/Output_Layer_Values.pdf", dpi=300)
plt.show()
