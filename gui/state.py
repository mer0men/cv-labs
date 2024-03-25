import cv2

class State:
    filters = ["rgb", "red", "green", "blue", "intensity", "brightness"]
    selected_filter = "rgb"

    def filter_image(self, image):
        if self.selected_filter == "rgb": 
            return image 
        if self.selected_filter == "red":
            filtered = image.copy()
            filtered[:, :, 1] = 0
            filtered[:, :, 2] = 0
            return filtered
        if self.selected_filter == "green":
            filtered = image.copy()
            filtered[:, :, 0] = 0
            filtered[:, :, 2] = 0
            return filtered
        if self.selected_filter == "blue":
            filtered = image.copy()
            filtered[:, :, 0] = 0
            filtered[:, :, 1] = 0
            return filtered
        if self.selected_filter == "intensity":
            filtered = image.copy() // 3
            return filtered
        if self.selected_filter == "brightness":
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

state = State()

