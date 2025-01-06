from .drawer import Drawer

class VideoSaver:
    def __init__(self):
        self.data = []
        self.drawer = Drawer()
    
    def draw(self, vision):
        pass

    def save_video(frames, fps=30):
        # Frames should be an iterable of ndarray
        import imageio
        imageio.mimsave('avideo.mp4', frames, fps=fps, bitrate=10 ** 7)
