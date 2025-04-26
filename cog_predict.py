# flake8: noqa
# This file is used for deploying replicate models

import os

os.system('pip install gfpgan')
os.system('python setup.py develop')

import cv2
import shutil
import tempfile
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan.utils import RealESRGANer

try:
    from cog import BasePredictor, Input, Path
    from gfpgan import GFPGANer
except Exception:
    print('please install cog and realesrgan package')


class Predictor(BasePredictor):

    def setup(self):
        os.makedirs('output', exist_ok=True)
        # download weights
        if not os.path.exists('weights/4x_span_pretrain.pth'):
            os.system(
                'wget https://github.com/tutope8/MODG147/raw/refs/heads/main/4x_span_pretrain.pth -P ./weights'
            )
        if not os.path.exists('weights/GFPGANv1.4.pth'):
            os.system('wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./weights')

    def choose_model(self, scale, tile=0):
        half = True if torch.cuda.is_available() else False
        # Configuración específica para el modelo SPAN
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model_path = 'weights/4x_span_pretrain.pth'
        self.upsampler = RealESRGANer(
            scale=4, 
            model_path=model_path, 
            model=model, 
            tile=tile, 
            tile_pad=10, 
            pre_pad=0, 
            half=half,
            strict_load_g=False  # Añadido para compatibilidad con SPAN
        )

        self.face_enhancer = GFPGANer(
            model_path='weights/GFPGANv1.4.pth',
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)

    def predict(
        self,
        img: Path = Input(description='Input'),
        scale: float = Input(description='Factor de reescalado', default=2),
        face_enhance: bool = Input(
            description='Mejorar rostros con GFPGAN. No funciona para imágenes/videos de anime', default=False),
        tile: int = Input(
            description=
            'Tamaño de mosaico. Por defecto es 0 (sin mosaico). Si hay problemas de memoria GPU, especificar un valor como 400 o 200',
            default=0)
    ) -> Path:
        if tile <= 100 or tile is None:
            tile = 0
        print(f'img: {img}. scale: {scale}. face_enhance: {face_enhance}. tile: {tile}.')
        try:
            extension = os.path.splitext(os.path.basename(str(img)))[1]
            img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            elif len(img.shape) == 2:
                img_mode = None
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_mode = None

            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            self.choose_model(scale, tile)

            try:
                if face_enhance:
                    _, _, output = self.face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = self.upsampler.enhance(img, outscale=scale)
            except RuntimeError as error:
                print('Error', error)
                print('Si encuentra un error de memoria CUDA, intente establecer "tile" a un tamaño menor, por ejemplo, 400.')

            if img_mode == 'RGBA':  # Las imágenes RGBA deben guardarse en formato png
                extension = 'png'
            out_path = Path(tempfile.mkdtemp()) / f'out.{extension}'
            cv2.imwrite(str(out_path), output)
        except Exception as error:
            print('excepción global: ', error)
        finally:
            clean_folder('output')
        return out_path


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
