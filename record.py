import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import wave
import wavio
import sys
form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 8192# 2^12 samples for buffer
dev_index = 1 # device index found by p.get_device_info_by_index(ii)
audio = pyaudio.PyAudio() # create pyaudio instantiation

print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

print("-------------------------------------------------------------")
# create pyaudio stream
existing_data = json.loads(open('demo.json').read())

person = input('Who is recording this data?')

if person not in existing_data:
    existing_data[person] = []
sample_count = int(input('How many samples?'))

recorded_data = existing_data[person]


for i in range(sample_count):
    input('Press enter to start recording')
    time.sleep(.1)
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans,input_device_index = dev_index,input = True, frames_per_buffer=chunk)
    # record data chunk
    stream.start_stream()
    data = np.fromstring(stream.read(chunk),dtype=np.int16)
    print(data)
    print('Done recording')
    stream.stop_stream()
    stream.close()


    wavio.write('wavioaudio.wav',data,samp_rate,sampwidth=1)
    wf = wave.open('testaudio.wav', 'w')
    wf.setnchannels(chans)
    wf.setsampwidth(audio.get_sample_size(form_1))
    wf.setframerate(samp_rate)
    print(data.tolist())
    wf.writeframes(data.tobytes())
    wf.close()
    # mic sensitivity correction and bit conversion
    mic_sens_dBV = -47.0 # mic sensitivity in dBV + any gain
    mic_sens_corr = np.power(10.0,mic_sens_dBV/20.0) # calculate mic sensitivity conversion factor

    # (USB=5V, so 15 bits are used (the 16th for negatives)) and the manufacturer microphone sensitivity corrections
    data = ((data/np.power(2.0,15))*5.25)*(mic_sens_corr)

    # compute FFT parameters
    f_vec = samp_rate*np.arange(chunk/2)/chunk # frequency vector based on window size and sample rate
    mic_low_freq = 100 # low frequency response of the mic (mine in this case is 100 Hz)
    low_freq_loc = np.argmin(np.abs(f_vec-mic_low_freq))
    fft_data = (np.abs(np.fft.fft(data))[0:int(np.floor(chunk/2))])/chunk
    fft_data[1:] = 2*fft_data[1:]

    max_loc = np.argmax(fft_data[low_freq_loc:])+low_freq_loc

    # plot
    plt.style.use('ggplot')
    plt.rcParams['font.size']=18
    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(111)
    plt.plot(f_vec,fft_data)
    print(f_vec,len(fft_data))
    ax.set_ylim([0,2*np.max(fft_data)])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [Pa]')
    ax.set_xscale('log')
    plt.grid(True)

    # max frequency resolution
    plt.annotate(r'$\Delta f_{max}$: %2.1f Hz' % (samp_rate/(2*chunk)),xy=(0.7,0.92),\
                 xycoords='figure fraction')

    # annotate peak frequency
    annot = ax.annotate('Freq: %2.1f'%(f_vec[max_loc]),xy=(f_vec[max_loc],fft_data[max_loc]),\
                        xycoords='data',xytext=(0,30),textcoords='offset points',\
                        arrowprops=dict(arrowstyle="->"),ha='center',va='bottom')

    plt.savefig(f'fft_{person}.png',dpi=300,facecolor='#FCFCFC')
    plt.show()

    record_bool = input('Would you like to record this sample? y/n')

    if record_bool == 'y':
        #frequency, amplitude
        recorded_data.append([f_vec.tolist(),fft_data.tolist()])

existing_data[person] = recorded_data


with open('demo.json', 'w') as outfile:
    json.dump(existing_data, outfile)
