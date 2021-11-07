from data import colorize_image as CI

class Colorize:
    def __init__(self, Xd, gpu_id):
        self.model = CI.ColorizeImageTorch(gpu_id, Xd=Xd)
        self.model.prep_net(path='./models/pytorch/caffemodel.pth')
        
        self.Xd = Xd

    def compute(self, gray, rgb):
        if rgb.shape[0] != self.Xd or rgb.shape[1] != self.Xd:
            raise Exception(f'Input not expected size - is {rgb.shape}, expected {(self.Xd, self.Xd, 3)}!')
        
        # Load the gray image.
        # Converts the gray image into an L component.
        self.model.set_image(gray)

        # Create AB channels from the low-res RGB.
        # These AB channels will be blended into the gray L component.
        im_lab = CI.rgb2lab_transpose(rgb)
        im_ab0 = im_lab[1:3, :, :]
        
        # Apply the blending.
        self.model.net_forward(im_ab0)

        # Construct the result.
        return self.model.get_img_fullres()[:, :, ::-1]

