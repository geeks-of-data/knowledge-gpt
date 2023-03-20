# Read the API key from the file
with open(r"C:\ToDo\Work\2023-1_RadarDataProject\key.cfg", "r") as f:
    SECRET_KEY = f.read().strip().split("=")[1]