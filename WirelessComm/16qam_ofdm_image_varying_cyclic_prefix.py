from PIL import Image
import os             # for working with arrays, fourier transforms, linear algebra and matrices
import matplotlib.pyplot as plt     # for visualization, to plot the data
import scipy.interpolate            # sub-package for objects used in interpolation

from plot import *                  # for plotting the images

np.random.seed(0)

# Defining the parameters
subCarriers = 64                    # common sub-carrier length
cyclicPrefix = subCarriers // 4     # 25% length of the subcarrier
pilot = 8                           # common pilot value
pilotVal = 3+3j                     # amp = 4.24, phase angle = pi/4 radians

allCarriers = np.arange(subCarriers)                        # indices of all subcarriers ([0, 1, ... subCarriers - 1])
pilotCarriers = allCarriers[::subCarriers//pilot]           # place pilot in every (subcarriers/pilot)th carrier

pilotCarriers = np.append(pilotCarriers, allCarriers[-1])   # adding last subcarrier as a pilot (for channel estimation)
pilot = pilot + 1                                           # update total num of pilots

dataCarriers = np.delete(allCarriers, pilotCarriers)        # exclude the pilot carriers from the data carriers

mu = 4
totalData_per_symbol = len(dataCarriers) * mu # number of bits that can be transmitted in a single OFDM symbol (224)
mapping_table = {                             # map all the bits with gray-mapping, ensuring that points differ by
    (0,0,0,0) : -3-3j,                        # only 1 bit are adjacent to each other in the constellation.
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

# ==> each OFDM symbol is subCarriers - pilot = 64 - 9 = 55 usable for data (dataCarriers)
# ==> each OFDM symbol is 55 * 4 = 220 bits/symbol


# Define the wireless channel between the Tx and Rx.
# To enhance the simulation's realism, the channel response is defined and add SNR.
channelResponse = np.array([1, 0, 0.3+0.3j])    # the impulse response of the wireless channel:
                                                # 1'st element - main signal path
                                                # 2'nd element - no signal received
                                                # 3'rd element - delayed path, amp - 0.424, phase shift - pi/4 radians
FFT_channelResponse = np.fft.fft(channelResponse, subCarriers) # FFT to channel response for channel estimation
SNR = 24                                        # signal-to-noise ratio in dB

# OFDM process
def s_to_p(bits):
    return bits.reshape((len(dataCarriers), mu))

def encoder(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])    # convert bit group according to the mapping table

def create_symbol(data):
    symbol = np.zeros(subCarriers, dtype=complex)
    symbol[pilotCarriers] = pilotVal
    symbol[dataCarriers] = data
    return symbol

def ifft(data_OFDM):
    return np.fft.ifft(data_OFDM)

def add_cp(time_OFDM):
    cp = time_OFDM[-cyclicPrefix:]
    return np.append(cp, time_OFDM)

def channel(signal):
    convolved = np.convolve(signal, channelResponse) # the signal at Rx is the convolution of the transmit signal
                                                     # with the channel response
    p_avg= np.mean(abs(convolved**2))            # no need to divide by 2 as the signal is complex-valued
    noise_power = p_avg * 10**(-SNR/10)          # noise power = σ^2
                                                 # using SNR ratio instead of N=kTB as the latter is less practical
    print("RX Signal power: %.4f. Noise power: %.4f" % (p_avg, noise_power))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    # noise explanation:
    # np.sqrt(sigma2/2) - noise_power, split equally between real and imaginary parts
    # np.random.randn(*convolved.shape) - generates real-valued Gaussian (normal) random numbers.
    # Mean is 0, and variance is 1.
    # *convolved.shape unpacks the shape of the convolved signal, ensuring the noise has the same dimensions.
    # 1j*np.random.randn(*convolved.shape) - generates the imaginary part of the noise.
    # It's another set of Gaussian random numbers, multiplied by 1j to make them imaginary.
    return convolved + noise

def remove_cp(signal):
    return signal[cyclicPrefix:(cyclicPrefix+subCarriers)]

def fft(recieve_NOCP):
    return np.fft.fft(recieve_NOCP)

def channel_estimate(freq_OFDM):
    pilots = freq_OFDM[pilotCarriers]
    pilot_estimates = pilots / pilotVal # By dividing the received pilot values by the known transmitted pilot values,
                                        # we get an estimate of the channel's effect on these subcarriers.
    channelEstimation_amp = np.interp(allCarriers, pilotCarriers, np.abs(pilot_estimates))     # absolute val
    channelEstimation_phase = np.interp(allCarriers, pilotCarriers, np.angle(pilot_estimates)) # phase
    full_estimate = channelEstimation_amp * np.exp(1j*channelEstimation_phase) # polar form
    
    return full_estimate, pilot_estimates

def domain_equalizer(freq_OFDM, estimated_signal, threshold):
    mask = np.abs(estimated_signal) < threshold # boolean mask identifying channel estimates below the threshold
    safe = np.where(mask, np.sign(estimated_signal) * threshold, estimated_signal) # replaces small values with the
                                                                                   # threshold value
    equalized = freq_OFDM / safe # the noise is not factored in the equalization for simplicity
    equalized[mask] = 0          # equalized values set to zero for subcarriers with very weak channel estimates,
                                 # can help prevent noise amplification.
    return equalized

def get_data(equalized_signal):
    return equalized_signal[dataCarriers]

decoding_table = {v : k for k, v in mapping_table.items()} # inverse mapping of the mapping table

def decoder(estimated_data):
    known_constellation = np.array([x for x in decoding_table.keys()])
    distance = abs(known_constellation.reshape(1,-1) - estimated_data.reshape(-1, 1)) # after reshaping, distance 2D:
                                                                                      # known_constellation (1, n)
                                                                                      # estimated_data (m,1)
    closest_index = distance.argmin(axis=1)
    closest_points = known_constellation[closest_index]

    return np.vstack([decoding_table[C] for C in closest_points]), closest_points # returns 2D array of closest_point bits
                                                                                # and the closest_point array

def p_to_s(bits):
    return bits.reshape(-1,)

# Load and preprocess image
def load_image(path, size=(28, 28)):
    img = Image.open(path).convert("L").resize(size)  # grayscale
    img_array = np.array(img)
    return img_array, img.size

# Convert image array to bitstream
def image_to_bits(img_array):
    flat = img_array.flatten()
    bits = np.unpackbits(flat.astype(np.uint8))
    return bits

# Convert bitstream to image array
def bits_to_image(bits, image_shape):
    bytes_array = np.packbits(bits)
    image_array = bytes_array.reshape(image_shape)
    # image_array = np.where(img_array > 127, 255, 0)
    return image_array

# Save reconstructed image
def save_image(img_array, path):
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(path)

# Prepare image and bitstream
image_path = "sample_image.png"  # replace with your actual image filename
img_array, img_shape = load_image(image_path)
image_bits = image_to_bits(img_array)

# Process in chunks
received_bits = []
chunk_size = totalData_per_symbol
for i in range(0, len(image_bits), chunk_size):
    bits_chunk = image_bits[i:i+chunk_size]
    if len(bits_chunk) < chunk_size:
        bits_chunk = np.pad(bits_chunk, (0, chunk_size - len(bits_chunk)))
    
    bits_SP = s_to_p(bits_chunk)
    qam_VAL = encoder(bits_SP)
    data_OFDM = create_symbol(qam_VAL)
    time_OFDM = ifft(data_OFDM)
    cp_OFDM = add_cp(time_OFDM)
    transmit_sig = cp_OFDM
    receive_sig = channel(transmit_sig)
    receive_NOCP = remove_cp(receive_sig)
    freq_OFDM = fft(receive_NOCP)
    estimated_signal, estimated_pilot = channel_estimate(freq_OFDM)
    threshold = np.mean(np.abs(estimated_signal)**2) * 0.01 # 1% of average channel
    equalized_signal = domain_equalizer(freq_OFDM, estimated_signal, threshold)
    estimated_data = get_data(equalized_signal)
    decoder_bits, _ = decoder(estimated_data)
    final_bits = p_to_s(decoder_bits)
    received_bits.extend(final_bits[:len(bits_chunk)])

# Convert received bits to image and save
received_bits = np.array(received_bits[:len(image_bits)])
reconstructed_image = bits_to_image(received_bits, img_array.shape)
save_image(reconstructed_image, "reconstructed_image.png")
print("Image transmission complete. Output saved as reconstructed_image.png.")

# ==========================
# Evaluation Under Varying Cyclic Prefix Lengths
# ==========================

def calculate_ber(original_bits, received_bits):
    return np.mean(original_bits != received_bits)

def calculate_ser(original_symbols, received_symbols):
    return np.mean(original_symbols != received_symbols)

def calculate_papr(signal):
    peak_power = np.max(np.abs(signal)**2)
    avg_power = np.mean(np.abs(signal)**2)
    return 10 * np.log10(peak_power / avg_power)

cyclic_prefix_options = [4, 8, 12, 16, 20, 24, 28, 32]  # Test different CP lengths
ber_cp_list = []
ser_cp_list = []
papr_cp_list = []

for cp_len in cyclic_prefix_options:
    cyclicPrefix = cp_len
    allCarriers = np.arange(subCarriers)
    pilotCarriers = allCarriers[::subCarriers // pilot]
    pilotCarriers = np.append(pilotCarriers, allCarriers[-1])
    pilot_count = len(pilotCarriers)
    dataCarriers = np.delete(allCarriers, pilotCarriers)
    totalData_per_symbol = len(dataCarriers) * mu

    received_bits_test = []
    original_bits_test = []
    ser_errors = 0
    total_symbols = 0
    papr_accumulator = []

    for i in range(0, len(image_bits), totalData_per_symbol):
        bits_chunk = image_bits[i:i+totalData_per_symbol]
        if len(bits_chunk) < totalData_per_symbol:
            bits_chunk = np.pad(bits_chunk, (0, totalData_per_symbol - len(bits_chunk)))

        bits_SP = s_to_p(bits_chunk)
        qam_VAL = encoder(bits_SP)
        data_OFDM = create_symbol(qam_VAL)
        time_OFDM = ifft(data_OFDM)
        cp_OFDM = add_cp(time_OFDM)
        transmit_sig = cp_OFDM
        papr_accumulator.append(calculate_papr(transmit_sig))
        receive_sig = channel(transmit_sig)
        receive_NOCP = remove_cp(receive_sig)
        freq_OFDM = fft(receive_NOCP)
        estimated_signal, estimated_pilot = channel_estimate(freq_OFDM)
        threshold = np.mean(np.abs(estimated_signal)**2) * 0.01
        equalized_signal = domain_equalizer(freq_OFDM, estimated_signal, threshold)
        estimated_data = get_data(equalized_signal)
        decoder_bits, closest_points = decoder(estimated_data)
        final_bits = p_to_s(decoder_bits)

        received_bits_test.extend(final_bits[:len(bits_chunk)])
        original_bits_test.extend(bits_chunk)
        ser_errors += np.sum(closest_points != qam_VAL)
        total_symbols += len(qam_VAL)

    ber = calculate_ber(np.array(original_bits_test), np.array(received_bits_test))
    ser = ser_errors / total_symbols
    avg_papr = np.mean(papr_accumulator)

    ber_cp_list.append(ber)
    ser_cp_list.append(ser)
    papr_cp_list.append(avg_papr)

# Plotting Evaluation for Cyclic Prefix Length
plt.figure()
plt.plot(cyclic_prefix_options, ber_cp_list, marker='o')
plt.title("BER vs Cyclic Prefix Length")
plt.xlabel("Cyclic Prefix Length")
plt.ylabel("Bit Error Rate")
plt.grid(True)
plt.savefig("16qam_ber_vs_cp.png")

plt.figure()
plt.plot(cyclic_prefix_options, ser_cp_list, marker='x', color='orange')
plt.title("SER vs Cyclic Prefix Length")
plt.xlabel("Cyclic Prefix Length")
plt.ylabel("Symbol Error Rate")
plt.grid(True)
plt.savefig("16qam_ser_vs_cp.png")

plt.figure()
plt.plot(cyclic_prefix_options, papr_cp_list, marker='s', color='green')
plt.title("PAPR vs Cyclic Prefix Length")
plt.xlabel("Cyclic Prefix Length")
plt.ylabel("Peak to Average Power Ratio (dB)")
plt.grid(True)
plt.savefig("16qam_papr_vs_cp.png")