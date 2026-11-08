from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import numpy as np
import moderngl            # utile surtout pour les annotations de type (ctx: moderngl.Context)
from PIL import Image      # enregistrement PNG

@dataclass
class Frame:
    images: list[np.ndarray]   # N × (H, W, 3) uint8, RGB, une par caméra
    depths: list[np.ndarray]   # N × (H, W) float32, z-buffer in [0,1], évaluation seulement
    Rs: list[np.ndarray]       # N × (3, 3) float, rotation caméra→monde
    Cs: list[np.ndarray]       # N × (3,) float, position caméra dans le monde (m)
    K: np.ndarray              # (3, 3) float, intrinsèques communes
    t: float                   # temps de simulation (s)
    

class StereoCapture:
    def __init__(self, ctx: moderngl.Context, scene, cams, W=640, H=480):
        self.fbos: list[moderngl.Framebuffer] = []
        self.depths: list[moderngl.Texture] = []        
        self.ctx=ctx
        self.W=W
        self.H=H
        self.scene = scene
        for _ in cams:
            color=ctx.renderbuffer((W, H)) ## alloue un boc mémoire de taille (W,H) pour le rendu couleur
            depth=ctx.depth_texture((W,H)) ## alloue un bloc mémoire de taille (W,H) pour le rendu profondeur
            fbo=ctx.framebuffer(color,depth) ## relie couleur et profondeur pour former l'image de taille (W,H)
            self.fbos.append(fbo)
            self.depths.append(depth)
            
    def grab(self, cams, b_mode,t):
        images = []
        depths = []
        Rs = []
        Cs = []
        for (fbo, depth, cam) in zip(self.fbos, self.depths, cams):
            fbo.use()
            fbo.clear(0.55, 0.7, 0.9) #remplir avec ciel bleu
            self.scene.render(cam, self.W/self.H, extra=self.scene.target_items(), building_mode=b_mode)
            rgb=np.frombuffer(fbo.read(components=3), np.uint8).reshape(self.H, self.W, 3)[::-1].copy()
            d=np.frombuffer(depth.read(),np.float32).reshape(self.H, self.W)[::-1].copy()
            images.append(rgb)
            depths.append(d)
            R = np.column_stack([cam.right, -cam.up, cam.forward])   # (3, 3), rotation caméra->monde
            C = np.array(cam.position, dtype=float)                   # (3,) vecteur position caméra dans repère monde
            Rs.append(R)
            Cs.append(C)
        self.ctx.screen.use()
        return Frame(images,depths,Rs,Cs,cams[0].intrinsics(self.W, self.H),t)
    
    @staticmethod
    def linearize_depth(d: np.ndarray, near: float, far: float)-> np.ndarray:
        z_ndc = 2*d - 1                             
        z = 2*near*far / (far + near - z_ndc*(far - near))
        return z
    