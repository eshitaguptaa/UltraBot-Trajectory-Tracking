# LesionPrediction.msg
# LSTM next-position prediction

float32 current_x      # current detection center x (pixels)
float32 current_y      # current detection center y (pixels)
float32 predicted_x    # LSTM predicted next x (pixels)
float32 predicted_y    # LSTM predicted next y (pixels)
float32 delta_x        # predicted velocity x (normalized)
float32 delta_y        # predicted velocity y (normalized)
int32 frame_id
bool lstm_ready        # True if LSTM has sufficient consecutive hits
int32 consecutive_hits
