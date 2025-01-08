from scripts import LoadTools
# Initialize LoadTools class
load_tools = LoadTools()

# Load YOLO model using alias
# model_path = load_tools.load_model("bat_trackingv4")

# Load YOLO model using .txt file path
model_path = load_tools.load_model("models/YOLO/bat_tracking/model_weights/bat_tracking.txt")

# hit distance is 400, exit-velocity 80, launch angle is 27