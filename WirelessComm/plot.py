import matplotlib.pyplot as plt
import numpy as np
from itertools import product       # to easily iterate the mapping table

def plotCarriers(pilotCarriers, dataCarriers, subCarriers, name):
    plt.figure(figsize=(8,1.5))
    plt.title("Carriers")
    plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label = 'pilot')
    plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label = 'data')
    plt.legend(fontsize = 10, ncol = 2)
    plt.xlim((-1, subCarriers))
    plt.ylim((-0.1, 0.3))
    plt.yticks([])
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name)

def plotConstellationMap(mapping_table, mu, name):
    plt.figure(figsize=(2**(mu // 2), 2**(mu // 2)))
    for B in product([0,1], repeat = mu):
        Q = mapping_table[B]
        plt.plot(Q.real, Q.imag, 'bo')
        plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha = 'center')
    plt.title("Constellation with Gray-Mapping")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.axis('equal')
    plt.xlim(- 2**(mu // 2), 2**(mu // 2))
    plt.ylim(- 2**(mu // 2), 2**(mu // 2))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name)


def plotChannelResponse(allCarriers, FFT_channelResponse, subCarriers, name):
    plt.figure(figsize=(6,4))
    plt.title("Channel Response")
    plt.plot(allCarriers, abs(FFT_channelResponse))
    plt.xlabel("Subcarrier index")
    plt.ylabel("$|H(f)|$")
    plt.xlim(0, subCarriers)
    plt.ylim(0.4, 1.6)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name)

def plotTransRecv(transmit_sig, receive_sig, name):
    plt.figure(figsize=(8,3))
    plt.title("OFDM Signals")
    plt.plot(abs(transmit_sig), label='Tx signal')
    plt.plot(abs(receive_sig), label='Rx signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time')  # the x-axis represents one sample of the OFDM signal, the time depends on the sampling rate
    plt.ylabel('$|signal|$')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name)

def plotChannelEstimation(allCarriers, FFT_channelResponse, pilotCarriers, pilot_estimates, full_estimate, name):
    plt.figure(figsize=(12,6))
    plt.title("Channel Estimation")
    plt.plot(allCarriers, abs(FFT_channelResponse), label='Correct channel')
    plt.stem(pilotCarriers, abs(pilot_estimates), label='Pilot estimates')
    plt.plot(allCarriers, abs(full_estimate), label='Estimated channel interpolation')
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Carrier index')
    plt.ylabel('$|H(f)|$')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name)

def plotRecvConstellation(estimated_data, mu, name):
    plt.figure(figsize=(2**(mu // 2), 2**(mu // 2)))
    plt.title("Received Constellation")
    plt.plot(estimated_data.real, estimated_data.imag, 'bo')
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.axis('equal')
    plt.xlim(- 2**(mu // 2), 2**(mu // 2))
    plt.ylim(- 2**(mu // 2), 2**(mu // 2))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name)

def plotDecodingPoints(estimated_data, closest_points, mu, name):
    plt.figure(figsize=(2**(mu // 2), 2**(mu // 2)))
    for estimate, closest in zip(estimated_data, closest_points):
        plt.plot([estimate.real, closest.real], [estimate.imag, closest.imag], 'b-o')
        plt.plot(closest_points.real, closest_points.imag, 'ro')

    plt.title("Constellation with Received Symbols")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.axis('equal')
    plt.xlim(- 2**(mu // 2), 2**(mu // 2))
    plt.ylim(- 2**(mu // 2), 2**(mu // 2))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name)