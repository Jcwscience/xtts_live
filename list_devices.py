import sounddevice as sd

# Get a list of available output devices

print(sd.query_devices())