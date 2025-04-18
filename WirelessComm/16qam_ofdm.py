import numpy as np                  # for working with arrays, fourier transforms, linear algebra and matrices
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

plotCarriers(   
    pilotCarriers=pilotCarriers,
    dataCarriers=dataCarriers,
    subCarriers=subCarriers,
    name="16qam_carriers.png"
)

plotConstellationMap(
    mapping_table=mapping_table,
    mu=mu,
    name="16qam_constellation_map.png"
)

# Define the wireless channel between the Tx and Rx.
# To enhance the simulation's realism, the channel response is defined and add SNR.
channelResponse = np.array([1, 0, 0.3+0.3j])    # the impulse response of the wireless channel:
                                                # 1'st element - main signal path
                                                # 2'nd element - no signal received
                                                # 3'rd element - delayed path, amp - 0.424, phase shift - pi/4 radians
FFT_channelResponse = np.fft.fft(channelResponse, subCarriers) # FFT to channel response for channel estimation
SNR = 24                                        # signal-to-noise ratio in dB

plotChannelResponse(
    allCarriers=allCarriers,
    FFT_channelResponse=FFT_channelResponse,
    subCarriers=subCarriers,
    name="16qam_channel_response.png"
)

# OFDM process
# Generating bits entering the modem and S/P (serial to parallel).
bits = np.random.binomial(n=1, p=0.5, size=totalData_per_symbol) # generate binary data using Bernoulli distribution
                                                                 # total num of data bits (excluding pilots) = 220
def s_to_p(bits):
    return bits.reshape((len(dataCarriers), mu))

bits_SP = s_to_p(bits)

# Encode the parallel bits using the mapping table.
# Also, allocate each subcarrier with data/pilot - create OFDM symbol
def encoder(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])    # convert bit group according to the mapping table

qam_VAL = encoder(bits_SP)
print(qam_VAL)
def create_symbol(data):
    symbol = np.zeros(subCarriers, dtype=complex)
    symbol[pilotCarriers] = pilotVal
    symbol[dataCarriers] = data
    return symbol

data_OFDM = create_symbol(qam_VAL)
print(data_OFDM)
# Transform the symbol to the time-domain using IFFT
def ifft(data_OFDM):
    return np.fft.ifft(data_OFDM)

time_OFDM = ifft(data_OFDM)

# Add cyclic prefix to the symbol
def add_cp(time_OFDM):
    cp = time_OFDM[-cyclicPrefix:]
    return np.append(cp, time_OFDM)

cp_OFDM = add_cp(time_OFDM)

# Define the wireless channel as a static multipath channel with impulse response
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

# Transmit and receive signal
transmit_sig = cp_OFDM
receive_sig = channel(transmit_sig)

plotTransRecv(
    transmit_sig=transmit_sig,
    receive_sig=receive_sig,
    name="16qam_signals.png"
)

# Remove the CP at Rx 
def remove_cp(signal):
    return signal[cyclicPrefix:(cyclicPrefix+subCarriers)]

receive_NOCP= remove_cp(receive_sig)

# Transform the symbol back to the frequency-domain using FFT
def fft(recieve_NOCP):
    return np.fft.fft(recieve_NOCP)

freq_OFDM = fft(receive_NOCP)

# Invert channel (frequency domain equalizer). 
# Channel estimation using zero-forcing (to mitigate ISI and co-channel interference)
# followed by linear interpolation.
def channel_estimate(freq_OFDM):
    pilots = freq_OFDM[pilotCarriers]
    pilot_estimates = pilots / pilotVal # By dividing the received pilot values by the known transmitted pilot values,
                                        # we get an estimate of the channel's effect on these subcarriers.
    channelEstimation_amp = np.interp(allCarriers, pilotCarriers, np.abs(pilot_estimates))     # absolute val
    channelEstimation_phase = np.interp(allCarriers, pilotCarriers, np.angle(pilot_estimates)) # phase
    full_estimate = channelEstimation_amp * np.exp(1j*channelEstimation_phase) # polar form
    
    return full_estimate, pilot_estimates

estimated_signal, estimated_pilot = channel_estimate(freq_OFDM)

plotChannelEstimation(
        allCarriers=allCarriers,
        FFT_channelResponse=FFT_channelResponse,
        pilotCarriers=pilotCarriers,
        pilot_estimates=estimated_pilot,
        full_estimate=estimated_signal,
        name="16qam_channel_estimation.png"
    )

def domain_equalizer(freq_OFDM, estimated_signal, threshold):
    mask = np.abs(estimated_signal) < threshold # boolean mask identifying channel estimates below the threshold
    safe = np.where(mask, np.sign(estimated_signal) * threshold, estimated_signal) # replaces small values with the
                                                                                   # threshold value
    equalized = freq_OFDM / safe # the noise is not factored in the equalization for simplicity
    equalized[mask] = 0          # equalized values set to zero for subcarriers with very weak channel estimates,
                                 # can help prevent noise amplification.
    return equalized

threshold = np.mean(np.abs(estimated_signal)**2) * 0.01 # 1% of average channel
equalized_signal = domain_equalizer(freq_OFDM, estimated_signal, threshold)

# Extract the data carriers from the equalized symbol.
def get_data(equalized_signal):
    return equalized_signal[dataCarriers]

estimated_data = get_data(equalized_signal)

plotRecvConstellation(
    estimated_data=estimated_data,
    mu=mu, 
    name="16qam_received_constellation.png"
)

# Decode the data. In order to do this, we compare each received constellation point against each possible 
# constellation point and choose the constellation point which is closest to the received point.

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
decoder_bits, closest_points = decoder(estimated_data)

plotDecodingPoints(
    estimated_data=estimated_data, 
    closest_points=closest_points,
    mu=mu, 
    name="16qam_constellation_decoding.png"
)

def p_to_s(bits):
    return bits.reshape(-1,)

final_bits = p_to_s(decoder_bits)
print("Bit error rate:", np.sum(abs(bits-final_bits)/len(bits))) # high SNR = lower chance of bit errors
