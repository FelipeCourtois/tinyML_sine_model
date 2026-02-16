"""
Real-time plotter for visualizing TinyML model predictions against true values.

This script reads serial data from a microcontroller
that is sending TinyML model inference results. It plots both the predicted
and the true sine wave values in real-time using Matplotlib.

The expected serial data format is a string like: "Pred:0.50,True:0.52"
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import re

# --- Configuration ---
# For WSL, use /dev/ttyACM0. For Windows, use a COM port like COM3.
SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE = 115200
MAX_POINTS = 200  # Number of points to display on the plot (window width)

# --- Data Storage ---
# Using deque for efficient appending and popping from both ends.
# It automatically discards old data when the maxlen is reached.
data_pred = deque(maxlen=MAX_POINTS)
data_true = deque(maxlen=MAX_POINTS)
data_x = deque(maxlen=MAX_POINTS)

# --- Thread Control ---
# Flag to control the execution of the serial reading thread.
running = True

# --- Serial Reading Function (runs in a separate thread) ---
def read_serial():
    """
    Reads data from the serial port in a separate thread.
    
    This function continuously reads lines from the specified serial port,
    parses the data to extract predicted and true values, and appends them
    to the global data deques.
    """
    global running
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}...")
        
        counter = 0
        while running:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # Regex to find and extract float or int values from the serial string.
            # Expected format: "Pred:0.50,True:0.52"
            match = re.search(r"Pred:([-+]?\d*\.\d+|[-+]?\d+),True:([-+]?\d*\.\d+|[-+]?\d+)", line)
            
            if match:
                pred_val = float(match.group(1))
                true_val = float(match.group(2))
                
                # Append new data to the deques
                data_pred.append(pred_val)
                data_true.append(true_val)
                data_x.append(counter)
                counter += 1
                
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        running = False

# --- Plot Configuration ---
fig, ax = plt.subplots(facecolor='#1e1e1e')  # Dark background for a sci-fi look
ax.set_facecolor('#1e1e1e')

# Line styles
line_true, = ax.plot([], [], color='#00ff00', label='True (Math)', linewidth=2, alpha=0.6)
line_pred, = ax.plot([], [], color='#ff00ff', label='Pred (TinyML)', linewidth=2, linestyle='--')

# Axes configuration
ax.set_ylim(-1.5, 1.5)  # Sine wave range is -1 to 1, add some margin
ax.set_title('TinyML Sine Wave Inference', color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', facecolor='#1e1e1e', labelcolor='white')

# Error text display
error_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='yellow', fontsize=10)

# --- Animation Function ---
def update(frame):
    """
    Update function for the Matplotlib animation.
    
    This function is called periodically to update the plot with new data.
    It redraws the lines, adjusts the x-axis to create a scrolling effect,
    and updates the error text.
    
    Args:
        frame: The current frame number (unused).
        
    Returns:
        A tuple of the artists that were updated.
    """
    if data_x:
        # Update line data
        line_pred.set_data(range(len(data_x)), data_pred)
        line_true.set_data(range(len(data_x)), data_true)
        
        # Adjust x-axis for a scrolling effect
        ax.set_xlim(0, len(data_x))
        
        # Calculate and display the current error
        current_error = abs(data_pred[-1] - data_true[-1])
        error_text.set_text(f"Current Error: {current_error:.4f}")

    return line_pred, line_true, error_text

# --- Main Execution ---
if __name__ == "__main__":
    # Start the serial reading thread
    serial_thread = threading.Thread(target=read_serial)
    serial_thread.daemon = True
    serial_thread.start()

    # Start the animation
    ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
    plt.show()

    # Stop the serial thread when the plot window is closed
    running = False
    serial_thread.join()