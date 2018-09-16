# -*- coding: utf-8 -*-
from pyaudio import PyAudio, paInt16 
import numpy as np 
from datetime import datetime 
import pylab as pl
import wave 
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import threading

#save data to filename.WAV
def save_wave_file(filename, data): 
    wf = wave.open(filename, 'wb') 
    wf.setnchannels(1) 
    wf.setsampwidth(2) 
    wf.setframerate(SAMPLING_RATE) 
    wf.writeframes("".join(data)) 
    wf.close() 


def test():
    NUM_SAMPLES = 50000     # Size of internally cached blocks
    P_NUM=300
    data_l=NUM_SAMPLES-P_NUM
    SAMPLING_RATE = 44100    # sampling rate
    COUNT_NUM = 0#20          

    # start recording
    print"record is  beginning!"
    pa = PyAudio()
    stream = pa.open(format=paInt16, channels=1, rate=SAMPLING_RATE, input=True, 
                    frames_per_buffer=NUM_SAMPLES) 

    save_count = 0 
    save_buffer = [] 

    while True: 
        # read the samples   NUM_SAMPLES
        string_audio_data = stream.read(NUM_SAMPLES) 
        # change the data to an array
        audio_data = np.fromstring(string_audio_data, dtype=np.short) 
        p_data=audio_data[P_NUM:NUM_SAMPLES]
        ap_data=p_data*signal.hann(data_l,sym=0)
        time=np.arange(0,NUM_SAMPLES-300)*(1.0/SAMPLING_RATE)
    
        fft_data=np.fft.rfft(p_data)/data_l
        afft_data=np.fft.rfft(ap_data)/data_l
        freqs=np.linspace(0,SAMPLING_RATE/2,data_l/2+1)
        xfft_data=20*np.log10(np.clip(np.abs(fft_data),1e-20,1e100))
        axfft_data=20*np.log10(np.clip(np.abs(afft_data),1e-20,1e100))
        z=np.argmax(np.abs(fft_data))
        n=(data_l*18000)/SAMPLING_RATE
        n1=(data_l*17600)/SAMPLING_RATE
        n2=(data_l*17950)/SAMPLING_RATE
        n3=(data_l*18050)/SAMPLING_RATE
        n4=(data_l*18400)/SAMPLING_RATE
        print n
        kfft_data=np.abs(fft_data)
        right=sum(kfft_data[n1:n2])
        left=sum(kfft_data[n3:n4])
        x=left-right
        print x
        if(abs(x)>80):
            
            if(left>right):
                print "your hand is moving toward left"
            
            else:
                print "your hand is moving toward right"
         
        else:
            print "No moving is detected"

        # plot
        pl.subplot(211)
        pl.plot(time,p_data)
        pl.subplot(212)
        pl.plot(freqs,np.abs(afft_data))
        pl.legend()
        pl.xlabel(u"频率(Hz)")
        pl.show()
        COUNT_NUM =COUNT_NUM +1;
        if COUNT_NUM >=1:
            break
def test1():
    fs = 44100 # Hz
    f = 18000 # Hz
    length = 1 #s
    myarray = np.arange(fs * length)
    myarray = np.sin(2 * np.pi * f / fs * myarray)
    sd.play(myarray, fs)

    
threading.Thread(target=test1).start()
threading.Thread(target= test).start()

    
