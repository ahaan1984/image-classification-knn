from PIL import Image
from typing import List, Tuple
import matplotlib.pyplot as plt

class ImageLoader:
    def __init__(self, path:str, width:int, height:int, show:bool=True) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.show = show

    def _load(self) -> Image:
        with Image.open(self.path) as img:
            if self.show:
                plt.imshow(img)
                plt.show()
            return img
        
    def grayscale(self, image:Image) -> Image:
        img = image.convert('L')
        if self.show:
            plt.imshow(img, cmap='gray')
            plt.show
        return img
    
    def resize(self, image:Image) -> Image:
        img = image.resize((self.width, self.height))
        if self.show:
            plt.imshow(img)
            plt.show()
        return img
    
    def flatten(self, image:Image) -> List[Tuple[int, int, int]]:
        return list(image.getdata())