from rp.core import RedPitayaBoard
from rp.constants import (
    RESET_INDEX_RP_DAC,
    RP_DAC_PORT_1,
    RP4_MAC,
    RP7_MAC,
    RP8_MAC,
    RAM_INIT_CONFIG_ID,
)
from rp.ram.config import RAM_SIZE, get_ram_config
from rp.adc.receive import AdcDataReceiver
from rp.misc.helpers import create_new_measure_folder
from helpers import (
    calculate_phase_difference,
    initDacBram,
    init_adc_sync,
    get_adc_samples,
)
from rp.adc.helpers import unpackADCData

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import time

PATH_TO_STORE_MASTER_MEASUREMENTS = "master_adc_measurements/"
PATH_TO_STORE_SLAVE_MEASUREMENTS = "slave_adc_measurements/"

VERBOSE = False

# Signal parameters
SIGNAL_TYPE = "Sine"
AMPLITUDE = 1.0
PHASE = 0
OFFSET = 0
START_V = 0.0
STOP_V = 1.0

# Dac parameters
DAC_SR_MHZ = 125
DWELL_TIME_MS = 1 / (DAC_SR_MHZ * 1000)

# RAM size
RAM_SIZE_SELECTED = RAM_SIZE.KB_64

N_WORKERS = 4

# ADC-Config
ADC_SR_MHZ = 125

if __name__ == "__main__":

    # create RedPitayaBoard objects
    MASTER_RP4 = RedPitayaBoard(mac=RP4_MAC)
    SLAVE_RP7 = RedPitayaBoard(mac=RP7_MAC)
    SLAVE_RP8 = RedPitayaBoard(mac=RP8_MAC)

    # send ram configs
    ram_config = get_ram_config(RAM_SIZE_SELECTED)
    MASTER_RP4.sendConfigParams(ram_config, RAM_INIT_CONFIG_ID)
    SLAVE_RP7.sendConfigParams(ram_config, RAM_INIT_CONFIG_ID)
    SLAVE_RP8.sendConfigParams(ram_config, RAM_INIT_CONFIG_ID)

    # Phase difference storage
    phase_diff = {
        "1st Slave": [],
        "2nd Slave": [],
    }
    
    dac_steps = list(range(1250, 140, -100)) + [125, 50, 40, 25, 20, 16, 10]
    SIG_FREQ_MHZ = [DAC_SR_MHZ / dac_step for dac_step in dac_steps]
    values = list(zip(dac_steps, SIG_FREQ_MHZ))

    # queue for storing results
    master_queue = mp.Queue()
    slave7_queue = mp.Queue()
    slave8_queue = mp.Queue()

    for dac_step, sig_freq_mhz in values:
         # Send DAC configuration for each device
        for device, device_mac in [("master", MASTER_RP4), ("slave7", SLAVE_RP7), ("slave8", SLAVE_RP8)]:
            initDacBram(
                pita=device_mac,
                signal_type=SIGNAL_TYPE,
                amplitude=AMPLITUDE,
                phase=PHASE,
                offset=OFFSET,
                start_V=START_V,
                stop_V=STOP_V,
                dac_steps=dac_step,
                dwell_time_ms=DWELL_TIME_MS,
                channel=RP_DAC_PORT_1,
                verbose=False
            )
        # ADC config for master and slaves
        adc_master_config = init_adc_sync(MASTER_RP4, DWELL_TIME_MS, ADC_SR_MHZ, ram_config, dac_step)
        init_adc_sync(SLAVE_RP7, DWELL_TIME_MS, ADC_SR_MHZ, ram_config, dac_step)
        init_adc_sync(SLAVE_RP8, DWELL_TIME_MS, ADC_SR_MHZ, ram_config, dac_step)


        MASTER_RP4.start_dac_sweep(RP_DAC_PORT_1)

        # send start sampling command
        MASTER_RP4.start_adc_sampling(adc_master_config["tcp"])
        SLAVE_RP7.start_adc_sampling(adc_master_config["tcp"])
        SLAVE_RP8.start_adc_sampling(adc_master_config["tcp"])

        print("Start sampling...")

        # get measurements from master and slaves
        master_proc = mp.Process(target=get_adc_samples, args=(MASTER_RP4, ram_config, master_queue))
        slave7_proc = mp.Process(target=get_adc_samples, args=(SLAVE_RP7, ram_config, slave7_queue))
        slave8_proc= mp.Process(target=get_adc_samples, args=(SLAVE_RP8, ram_config, slave8_queue))

        master_proc.start()
        slave7_proc.start()
        slave8_proc.start()

        master_proc.join()
        slave7_proc.join()
        slave8_proc.join()


        # Initialize lists to hold multiple data items
        master_data_list = []
        slave7_data_list = []
        slave8_data_list = []

        # Retrieve all data from master_queue
        while not master_queue.empty():
            master_data_list.append(master_queue.get())

        # Retrieve all data from slave7_queue
        while not slave7_queue.empty():
            slave7_data_list.append(slave7_queue.get())

        # Retrieve all data from slave8_queue
        while not slave8_queue.empty():
            slave8_data_list.append(slave8_queue.get())

        # Now process the data from each list
        master_ch1_list = []
        slave7_ch1_list = []
        slave8_ch1_list = []

        # Process each item from master_data_list
        for master_data in master_data_list:
            if master_data is not None:
                master_ch1 = unpackADCData(np.frombuffer(master_data, dtype=np.uint32), MASTER_RP4.id, rawData=False)[0]
                master_ch1_list.append(master_ch1)
            else:
                print("No data in master queue")

        # Process each item from slave7_data_list
        for slave7_data in slave7_data_list:
            if slave7_data is not None:
                slave7_ch1 = unpackADCData(np.frombuffer(slave7_data, dtype=np.uint32), SLAVE_RP7.id, rawData=False)[0]
                slave7_ch1_list.append(slave7_ch1)
            else:
                print("No data in slave7 queue")

        # Process each item from slave8_data_list
        for slave8_data in slave8_data_list:
            if slave8_data is not None:
                slave8_ch1 = unpackADCData(np.frombuffer(slave8_data, dtype=np.uint32), SLAVE_RP8.id, rawData=False)[0]
                slave8_ch1_list.append(slave8_ch1)
            else:
                print("No data in slave8 queue")

        print("Received all data from master and slaves...")

        # Stop DAC sweep after collecting all data
        MASTER_RP4.stop_dac_sweep()
        SLAVE_RP7.stop_dac_sweep()
        SLAVE_RP8.stop_dac_sweep()

        # Here, assume you want to calculate the phase difference for each received set of samples
        for master_ch1, slave7_ch1, slave8_ch1 in zip(master_ch1_list, slave7_ch1_list, slave8_ch1_list):
            # Calculate phase difference for each data set
            phase_diff["1st Slave"].append(calculate_phase_difference(master_ch1, slave7_ch1, sig_freq_mhz))
            phase_diff["2nd Slave"].append(calculate_phase_difference(master_ch1, slave8_ch1, sig_freq_mhz))


    # disable dac modules on master and slave
    MASTER_RP4.disable_module(RESET_INDEX_RP_DAC)
    SLAVE_RP7.disable_module(RESET_INDEX_RP_DAC)
    SLAVE_RP8.disable_module(RESET_INDEX_RP_DAC)

    # exit server application
    # MASTER_RP4.exitApplication()
    # SLAVE_RP7.exitApplication()
    # SLAVE_RP8.exitApplication()

    # close connections
    MASTER_RP4.close()
    SLAVE_RP7.close()
    SLAVE_RP8.close()

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(SIG_FREQ_MHZ, phase_diff["1st Slave"], marker = ".")
    ax[0].set_title("Phase difference between master and 1st slave")
    ax[0].set_xlabel("Signal frequency (MHz)")
    ax[0].set_ylabel("Phase difference (µs)")
    ax[0].grid()

    ax[1].plot(SIG_FREQ_MHZ, phase_diff["2nd Slave"], marker = ".")
    ax[1].set_title("Phase difference between master and 2nd slave")
    ax[1].set_xlabel("Signal frequency (MHz)")
    ax[1].set_ylabel("Phase difference (µs)")
    ax[1].grid()
    
    plt.tight_layout()  
    plt.show()




