import os
import io
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageChops
import tkinter as tk
from tkinter import ttk, Scale, IntVar, DoubleVar, StringVar, BooleanVar

# Try to import OpenCV for additional features
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class ImageEnhancer:
    """
    A class for enhancing images with various adjustments and filters.
    Provides basic adjustments (brightness, contrast, etc.) and advanced filters.
    """
    
    def __init__(self, controller=None):
        """
        Initialize the image enhancer.
        
        Args:
            controller: The controller object that handles the application logic
        """
        self.controller = controller
        
        # Default settings
        self.settings = {
            # Basic adjustments
            'brightness': 100,  # 0-200 (100 is original)
            'contrast': 100,    # 0-200 (100 is original)
            'saturation': 100,  # 0-200 (100 is original)
            'sharpness': 100,   # 0-200 (100 is original)
            'gamma': 100,       # 1-200 (100 is original)
            
            # Color adjustments
            'temperature': 0,   # -100 to 100 (0 is original)
            'tint': 0,          # -100 to 100 (0 is original)
            'vibrance': 0,      # 0-100
            
            # Advanced adjustments
            'highlights': 0,    # -100 to 100
            'shadows': 0,       # -100 to 100
            'clarity': 0,       # -100 to 100
            'dehaze': 0,        # 0-100
            
            # Filters
            'filter_type': 'none',  # none, grayscale, sepia, etc.
            'filter_amount': 100,   # 0-100
            
            # Noise reduction
            'noise_reduction': 0,   # 0-100
            
            # Vignette
            'vignette': 0,          # 0-100
            'vignette_color': '#000000',
            
            # Auto adjustments
            'auto_contrast': False,
            'auto_color': False,
            'auto_tone': False
        }
        
        # History for undo/redo
        self.history = []
        self.history_position = -1
        self.max_history = 10
    
    def enhance_image(self, image, **kwargs):
        """
        Enhance an image with various adjustments.
        
        Args:
            image: PIL Image object
            **kwargs: Additional settings to override defaults
            
        Returns:
            Enhanced PIL Image
        """
        # Update settings with any provided kwargs
        settings = self.settings.copy()
        settings.update(kwargs)
        
        # Make a copy of the image to avoid modifying the original
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        enhanced = image.copy()
        
        # Apply auto adjustments first if enabled
        if settings['auto_contrast']:
            enhanced = ImageOps.autocontrast(enhanced)
        
        if settings['auto_color']:
            # Auto color balance (equalize each channel separately)
            r, g, b, a = enhanced.split()
            r = ImageOps.equalize(r)
            g = ImageOps.equalize(g)
            b = ImageOps.equalize(b)
            enhanced = Image.merge('RGBA', (r, g, b, a))
        
        if settings['auto_tone']:
            # Auto tone (equalize luminance)
            r, g, b, a = enhanced.split()
            gray = ImageOps.grayscale(Image.merge('RGB', (r, g, b)))
            gray = ImageOps.equalize(gray)
            # Apply equalized luminance while preserving color
            enhanced = self._apply_luminance(enhanced, gray)
        
        # Apply basic adjustments
        enhanced = self._apply_basic_adjustments(enhanced, settings)
        
        # Apply color adjustments
        enhanced = self._apply_color_adjustments(enhanced, settings)
        
        # Apply advanced adjustments
        enhanced = self._apply_advanced_adjustments(enhanced, settings)
        
        # Apply filters
        enhanced = self._apply_filter(enhanced, settings)
        
        # Apply noise reduction
        enhanced = self._apply_noise_reduction(enhanced, settings)
        
        # Apply vignette
        enhanced = self._apply_vignette(enhanced, settings)
        
        return enhanced
    
    def _apply_basic_adjustments(self, image, settings):
        """Apply basic image adjustments."""
        # Extract alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # Apply brightness
        brightness = settings['brightness'] / 100.0
        if brightness != 1.0:
            rgb_image = ImageEnhance.Brightness(rgb_image).enhance(brightness)
        
        # Apply contrast
        contrast = settings['contrast'] / 100.0
        if contrast != 1.0:
            rgb_image = ImageEnhance.Contrast(rgb_image).enhance(contrast)
        
        # Apply saturation
        saturation = settings['saturation'] / 100.0
        if saturation != 1.0:
            rgb_image = ImageEnhance.Color(rgb_image).enhance(saturation)
        
        # Apply sharpness
        sharpness = settings['sharpness'] / 100.0
        if sharpness != 1.0:
            rgb_image = ImageEnhance.Sharpness(rgb_image).enhance(sharpness)
        
        # Apply gamma correction
        gamma = settings['gamma'] / 100.0
        if gamma != 1.0:
            # Convert to numpy array for faster processing
            if OPENCV_AVAILABLE:
                img_array = np.array(rgb_image)
                # Apply gamma correction
                img_array = np.power(img_array / 255.0, 1.0 / gamma) * 255.0
                # Convert back to uint8
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                rgb_image = Image.fromarray(img_array)
            else:
                # Use PIL for gamma correction
                gamma_table = [int(255 * ((i / 255) ** (1 / gamma))) for i in range(256)]
                rgb_image = rgb_image.point(lambda i: gamma_table[i])
        
        # Merge back with alpha channel
        return Image.merge('RGBA', (*rgb_image.split(), a))
    
    def _apply_color_adjustments(self, image, settings):
        """Apply color adjustments like temperature and tint."""
        # Extract alpha channel
        r, g, b, a = image.split()
        
        # Apply temperature adjustment
        temperature = settings['temperature']
        if temperature != 0:
            # Temperature shifts red and blue channels
            # Positive values make the image warmer (more red, less blue)
            # Negative values make the image cooler (less red, more blue)
            temp_factor = abs(temperature) / 100.0 * 30  # Scale factor
            
            if temperature > 0:
                # Warmer: increase red, decrease blue
                r = r.point(lambda i: min(255, i + temp_factor))
                b = b.point(lambda i: max(0, i - temp_factor))
            else:
                # Cooler: decrease red, increase blue
                r = r.point(lambda i: max(0, i - temp_factor))
                b = b.point(lambda i: min(255, i + temp_factor))
        
        # Apply tint adjustment
        tint = settings['tint']
        if tint != 0:
            # Tint shifts green and magenta
            # Positive values add green tint
            # Negative values add magenta tint (reduce green)
            tint_factor = abs(tint) / 100.0 * 30  # Scale factor
            
            if tint > 0:
                # Green tint: increase green
                g = g.point(lambda i: min(255, i + tint_factor))
            else:
                # Magenta tint: decrease green
                g = g.point(lambda i: max(0, i - tint_factor))
        
        # Apply vibrance
        vibrance = settings['vibrance']
        if vibrance != 0:
            # Vibrance increases saturation of less saturated colors more than already saturated ones
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(Image.merge('RGB', (r, g, b)))
                
                # Convert to HSV
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                # Calculate saturation mask (less saturated pixels get more adjustment)
                s = hsv[:, :, 1] / 255.0
                mask = (1.0 - s) * (vibrance / 100.0)
                
                # Apply vibrance
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + mask), 0, 255)
                
                # Convert back to RGB
                img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                
                # Convert back to PIL
                r, g, b = Image.fromarray(img_array).split()
            else:
                # Simplified vibrance using PIL
                # Increase saturation based on average saturation
                rgb_image = Image.merge('RGB', (r, g, b))
                enhanced = ImageEnhance.Color(rgb_image).enhance(1.0 + vibrance / 100.0)
                r, g, b = enhanced.split()
        
        # Merge back with alpha channel
        return Image.merge('RGBA', (r, g, b, a))
    
    def _apply_advanced_adjustments(self, image, settings):
        """Apply advanced adjustments like highlights, shadows, clarity, and dehaze."""
        # Extract alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # Apply highlights adjustment
        highlights = settings['highlights']
        if highlights != 0:
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(rgb_image)
                
                # Convert to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
                
                # Adjust L channel for highlights
                l_channel = lab[:, :, 0]
                
                # Create a mask for highlights (bright areas)
                highlight_mask = np.clip((l_channel - 128) / 127, 0, 1)
                
                # Apply adjustment
                factor = highlights / 100.0
                l_channel = l_channel + (highlight_mask * factor * 30)
                
                # Clip values
                lab[:, :, 0] = np.clip(l_channel, 0, 255)
                
                # Convert back to RGB
                img_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
                
                # Convert back to PIL
                rgb_image = Image.fromarray(img_array)
            else:
                # Simplified highlights adjustment using PIL
                # Create a mask for bright areas
                gray = ImageOps.grayscale(rgb_image)
                if highlights > 0:
                    # Brighten highlights
                    mask = gray.point(lambda i: (i / 255.0) ** 2 * 255)
                    bright = ImageEnhance.Brightness(rgb_image).enhance(1.0 + highlights / 100.0)
                    rgb_image = Image.composite(bright, rgb_image, mask)
                else:
                    # Darken highlights
                    mask = gray.point(lambda i: (i / 255.0) ** 2 * 255)
                    dark = ImageEnhance.Brightness(rgb_image).enhance(1.0 + highlights / 100.0)
                    rgb_image = Image.composite(dark, rgb_image, mask)
        
        # Apply shadows adjustment
        shadows = settings['shadows']
        if shadows != 0:
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(rgb_image)
                
                # Convert to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
                
                # Adjust L channel for shadows
                l_channel = lab[:, :, 0]
                
                # Create a mask for shadows (dark areas)
                shadow_mask = np.clip(1.0 - (l_channel / 128), 0, 1)
                
                # Apply adjustment
                factor = shadows / 100.0
                l_channel = l_channel + (shadow_mask * factor * 30)
                
                # Clip values
                lab[:, :, 0] = np.clip(l_channel, 0, 255)
                
                # Convert back to RGB
                img_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
                
                # Convert back to PIL
                rgb_image = Image.fromarray(img_array)
            else:
                # Simplified shadows adjustment using PIL
                # Create a mask for dark areas
                gray = ImageOps.grayscale(rgb_image)
                if shadows > 0:
                    # Brighten shadows
                    mask = gray.point(lambda i: ((255 - i) / 255.0) ** 2 * 255)
                    bright = ImageEnhance.Brightness(rgb_image).enhance(1.0 + shadows / 100.0)
                    rgb_image = Image.composite(bright, rgb_image, mask)
                else:
                    # Darken shadows
                    mask = gray.point(lambda i: ((255 - i) / 255.0) ** 2 * 255)
                    dark = ImageEnhance.Brightness(rgb_image).enhance(1.0 + shadows / 100.0)
                    rgb_image = Image.composite(dark, rgb_image, mask)
        
        # Apply clarity
        clarity = settings['clarity']
        if clarity != 0:
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(rgb_image)
                
                # Apply unsharp mask for clarity
                factor = abs(clarity) / 100.0
                radius = 10  # Larger radius for clarity vs sharpness
                amount = factor * 2.0
                threshold = 0
                
                # Create blurred version
                blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
                
                if clarity > 0:
                    # Enhance local contrast
                    sharpened = cv2.addWeighted(img_array, 1.0 + amount, blurred, -amount, 0)
                else:
                    # Reduce local contrast
                    sharpened = cv2.addWeighted(img_array, 1.0 - amount, blurred, amount, 0)
                
                # Convert back to PIL
                rgb_image = Image.fromarray(sharpened)
            else:
                # Simplified clarity using PIL's unsharp mask
                if clarity > 0:
                    # Enhance local contrast
                    radius = 10
                    percent = clarity
                    threshold = 0
                    rgb_image = rgb_image.filter(
                        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
                    )
                else:
                    # Reduce local contrast (blur slightly)
                    rgb_image = rgb_image.filter(
                        ImageFilter.GaussianBlur(radius=abs(clarity) / 50.0)
                    )
        
        # Apply dehaze
        dehaze = settings['dehaze']
        if dehaze > 0:
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(rgb_image)
                
                # Simple dehaze by increasing contrast and brightness
                factor = dehaze / 100.0
                
                # Increase contrast
                contrast_factor = 1.0 + factor * 0.5
                mean = np.mean(img_array, axis=(0, 1))
                img_array = (img_array - mean) * contrast_factor + mean
                
                # Increase brightness slightly
                brightness_factor = 1.0 + factor * 0.2
                img_array = img_array * brightness_factor
                
                # Clip values
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                # Convert back to PIL
                rgb_image = Image.fromarray(img_array)
            else:
                # Simplified dehaze using PIL
                # Increase contrast and brightness
                factor = dehaze / 100.0
                rgb_image = ImageEnhance.Contrast(rgb_image).enhance(1.0 + factor * 0.5)
                rgb_image = ImageEnhance.Brightness(rgb_image).enhance(1.0 + factor * 0.2)
        
        # Merge back with alpha channel
        r, g, b = rgb_image.split()
        return Image.merge('RGBA', (r, g, b, a))
    
    def _apply_filter(self, image, settings):
        """Apply creative filters to the image."""
        filter_type = settings['filter_type']
        filter_amount = settings['filter_amount'] / 100.0
        
        if filter_type == 'none' or filter_amount == 0:
            return image
        
        # Extract alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        if filter_type == 'grayscale':
            # Convert to grayscale
            gray_image = ImageOps.grayscale(rgb_image)
            gray_image = Image.merge('RGB', (gray_image, gray_image, gray_image))
            
            # Blend with original based on amount
            if filter_amount < 1.0:
                rgb_image = Image.blend(rgb_image, gray_image, filter_amount)
            else:
                rgb_image = gray_image
                
        elif filter_type == 'sepia':
            # Apply sepia tone
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(rgb_image)
                
                # Create sepia kernel
                sepia_kernel = np.array([
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131]
                ])
                
                # Apply sepia transformation
                sepia_array = cv2.transform(img_array, sepia_kernel)
                sepia_array = np.clip(sepia_array, 0, 255).astype(np.uint8)
                
                # Convert back to PIL
                sepia_image = Image.fromarray(sepia_array)
                
                # Blend with original based on amount
                if filter_amount < 1.0:
                    rgb_image = Image.blend(rgb_image, sepia_image, filter_amount)
                else:
                    rgb_image = sepia_image
            else:
                # Simplified sepia using PIL
                gray_image = ImageOps.grayscale(rgb_image)
                
                # Create sepia tone
                sepia_image = Image.merge('RGB', [
                    gray_image.point(lambda i: min(255, i * 1.2)),
                    gray_image.point(lambda i: min(255, i * 0.9)),
                    gray_image.point(lambda i: min(255, i * 0.7))
                ])
                
                # Blend with original based on amount
                if filter_amount < 1.0:
                    rgb_image = Image.blend(rgb_image, sepia_image, filter_amount)
                else:
                    rgb_image = sepia_image

        elif filter_type == 'negative':
            # Invert colors
            negative_image = ImageOps.invert(rgb_image)
            
            # Blend with original based on amount
            if filter_amount < 1.0:
                rgb_image = Image.blend(rgb_image, negative_image, filter_amount)
            else:
                rgb_image = negative_image
                
        elif filter_type == 'vintage':
            # Apply vintage effect (sepia + vignette + grain)
            # First apply sepia
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(rgb_image)
                
                # Create vintage color transformation
                vintage_kernel = np.array([
                    [0.6, 0.8, 0.2],
                    [0.2, 0.7, 0.1],
                    [0.2, 0.5, 0.5]
                ])
                
                # Apply vintage transformation
                vintage_array = cv2.transform(img_array, vintage_kernel)
                
                # Add slight yellow tint
                vintage_array[:, :, 0] = np.clip(vintage_array[:, :, 0] * 0.9, 0, 255)  # Reduce blue
                vintage_array[:, :, 1] = np.clip(vintage_array[:, :, 1] * 1.1, 0, 255)  # Boost green
                vintage_array[:, :, 2] = np.clip(vintage_array[:, :, 2] * 1.2, 0, 255)  # Boost red
                
                # Add grain
                grain = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
                vintage_array = cv2.addWeighted(vintage_array, 0.9, grain, 0.1, 0)
                
                # Add vignette
                rows, cols = vintage_array.shape[:2]
                
                # Generate vignette mask
                x = np.linspace(-1, 1, cols)
                y = np.linspace(-1, 1, rows)
                x_grid, y_grid = np.meshgrid(x, y)
                radius_grid = np.sqrt(x_grid**2 + y_grid**2)
                vignette = np.clip(1.0 - radius_grid, 0, 1)
                
                # Apply vignette
                for i in range(3):
                    vintage_array[:, :, i] = vintage_array[:, :, i] * vignette
                
                # Convert back to PIL
                vintage_image = Image.fromarray(vintage_array.astype(np.uint8))
                
                # Blend with original based on amount
                if filter_amount < 1.0:
                    rgb_image = Image.blend(rgb_image, vintage_image, filter_amount)
                else:
                    rgb_image = vintage_image
            else:
                # Simplified vintage using PIL
                # Apply sepia
                gray_image = ImageOps.grayscale(rgb_image)
                sepia_image = Image.merge('RGB', [
                    gray_image.point(lambda i: min(255, i * 1.2)),
                    gray_image.point(lambda i: min(255, i * 0.9)),
                    gray_image.point(lambda i: min(255, i * 0.7))
                ])
                
                # Add vignette
                width, height = sepia_image.size
                vignette = Image.new('RGB', (width, height), (0, 0, 0))
                radius = min(width, height) // 2
                gradient = Image.new('L', (width, height))
                draw = ImageDraw.Draw(gradient)
                
                for i in range(min(width, height) // 2):
                    alpha = 255 - i * 255 // radius
                    draw.ellipse(
                        [(width//2 - i, height//2 - i), (width//2 + i, height//2 + i)],
                        fill=alpha
                    )
                
                # Apply vignette
                vintage_image = Image.composite(sepia_image, vignette, gradient)
                
                # Blend with original based on amount
                if filter_amount < 1.0:
                    rgb_image = Image.blend(rgb_image, vintage_image, filter_amount)
                else:
                    rgb_image = vintage_image
                
        elif filter_type == 'cool':
            # Apply cool tone (blue tint)
            cool_image = rgb_image.copy()
            r, g, b = cool_image.split()
            
            # Increase blue, decrease red
            r = r.point(lambda i: max(0, i - 30 * filter_amount))
            b = b.point(lambda i: min(255, i + 30 * filter_amount))
            
            cool_image = Image.merge('RGB', (r, g, b))
            
            # Blend with original based on amount
            if filter_amount < 1.0:
                rgb_image = Image.blend(rgb_image, cool_image, filter_amount)
            else:
                rgb_image = cool_image
                
        elif filter_type == 'warm':
            # Apply warm tone (orange/yellow tint)
            warm_image = rgb_image.copy()
            r, g, b = warm_image.split()
            
            # Increase red and green, decrease blue
            r = r.point(lambda i: min(255, i + 30 * filter_amount))
            g = g.point(lambda i: min(255, i + 15 * filter_amount))
            b = b.point(lambda i: max(0, i - 30 * filter_amount))
            
            warm_image = Image.merge('RGB', (r, g, b))
            
            # Blend with original based on amount
            if filter_amount < 1.0:
                rgb_image = Image.blend(rgb_image, warm_image, filter_amount)
            else:
                rgb_image = warm_image
                
        elif filter_type == 'dramatic':
            # Apply dramatic effect (high contrast, dark shadows)
            if OPENCV_AVAILABLE:
                # Convert to OpenCV format
                img_array = np.array(rgb_image)
                
                # Convert to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
                
                # Increase contrast in L channel
                l_channel = lab[:, :, 0]
                mean_l = np.mean(l_channel)
                l_channel = (l_channel - mean_l) * (1.0 + filter_amount * 1.5) + mean_l
                
                # Darken shadows
                shadow_mask = np.clip(1.0 - (l_channel / 128), 0, 1)
                l_channel = l_channel - (shadow_mask * filter_amount * 20)
                
                # Clip values
                lab[:, :, 0] = np.clip(l_channel, 0, 255)
                
                # Increase color saturation
                lab[:, :, 1] = np.clip(lab[:, :, 1] * (1.0 + filter_amount * 0.5), 0, 255)
                lab[:, :, 2] = np.clip(lab[:, :, 2] * (1.0 + filter_amount * 0.5), 0, 255)
                
                # Convert back to RGB
                dramatic_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
                
                # Convert back to PIL
                dramatic_image = Image.fromarray(dramatic_array)
                
                # Blend with original based on amount
                if filter_amount < 1.0:
                    rgb_image = Image.blend(rgb_image, dramatic_image, filter_amount)
                else:
                    rgb_image = dramatic_image
            else:
                # Simplified dramatic effect using PIL
                # Increase contrast
                dramatic_image = ImageEnhance.Contrast(rgb_image).enhance(1.0 + filter_amount * 1.5)
                
                # Darken shadows
                shadow_mask = ImageOps.grayscale(rgb_image).point(
                    lambda i: ((255 - i) / 255.0) ** 2 * 255
                )
                darker = ImageEnhance.Brightness(dramatic_image).enhance(0.7)
                dramatic_image = Image.composite(darker, dramatic_image, shadow_mask)
                
                # Increase saturation
                dramatic_image = ImageEnhance.Color(dramatic_image).enhance(1.0 + filter_amount * 0.5)
                
                # Blend with original based on amount
                if filter_amount < 1.0:
                    rgb_image = Image.blend(rgb_image, dramatic_image, filter_amount)
                else:
                    rgb_image = dramatic_image
        
        # Merge back with alpha channel
        r, g, b = rgb_image.split()
        return Image.merge('RGBA', (r, g, b, a))
    
    def _apply_noise_reduction(self, image, settings):
        """Apply noise reduction to the image."""
        noise_reduction = settings['noise_reduction']
        
        if noise_reduction == 0:
            return image
        
        # Extract alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        if OPENCV_AVAILABLE:
            # Convert to OpenCV format
            img_array = np.array(rgb_image)
            
            # Apply noise reduction
            strength = noise_reduction / 100.0 * 15  # Scale factor
            
            # Non-local means denoising
            if noise_reduction < 30:
                # Light noise reduction
                denoised = cv2.fastNlMeansDenoisingColored(
                    img_array, None, strength, strength, 7, 21
                )
            else:
                # Stronger noise reduction
                denoised = cv2.bilateralFilter(
                    img_array, 9, strength * 2, strength * 2
                )
            
            # Convert back to PIL
            rgb_image = Image.fromarray(denoised)
        else:
            # Simplified noise reduction using PIL
            # Apply median filter for noise reduction
            if noise_reduction < 30:
                # Light noise reduction
                rgb_image = rgb_image.filter(ImageFilter.MedianFilter(3))
            elif noise_reduction < 70:
                # Medium noise reduction
                rgb_image = rgb_image.filter(ImageFilter.MedianFilter(5))
            else:
                # Strong noise reduction
                rgb_image = rgb_image.filter(ImageFilter.MedianFilter(7))
            
            # Apply slight blur to smooth out remaining noise
            rgb_image = rgb_image.filter(
                ImageFilter.GaussianBlur(radius=noise_reduction / 200.0)
            )
        
        # Merge back with alpha channel
        r, g, b = rgb_image.split()
        return Image.merge('RGBA', (r, g, b, a))
    
    def _apply_vignette(self, image, settings):
        """Apply vignette effect to the image."""
        vignette_amount = settings['vignette']
        
        if vignette_amount == 0:
            return image
        
        # Get vignette color
        vignette_color = self._parse_color(settings['vignette_color'])
        
        # Extract alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        width, height = image.size
        
        if OPENCV_AVAILABLE:
            # Convert to OpenCV format
            img_array = np.array(rgb_image)
            
            # Generate vignette mask
            rows, cols = img_array.shape[:2]
            
            # Create coordinate grids
            x = np.linspace(-1, 1, cols)
            y = np.linspace(-1, 1, rows)
            x_grid, y_grid = np.meshgrid(x, y)
            
            # Calculate radial distance from center
            radius_grid = np.sqrt(x_grid**2 + y_grid**2)
            
            # Create vignette mask
            vignette_mask = np.clip(1.0 - radius_grid * (vignette_amount / 100.0 * 1.5), 0, 1)
            
            # Create vignette color array
            vignette_array = np.ones_like(img_array) * np.array(vignette_color[:3])
            
            # Apply vignette
            for i in range(3):
                img_array[:, :, i] = img_array[:, :, i] * vignette_mask + \
                                     vignette_array[:, :, i] * (1 - vignette_mask)
            
            # Convert back to PIL
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
        else:
            # Simplified vignette using PIL
            # Create a radial gradient mask
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            # Calculate radii
            max_radius = math.sqrt(width**2 + height**2) / 2
            inner_radius = max_radius * (1 - vignette_amount / 100.0 * 0.8)
            
            # Draw gradient circles
            for i in range(int(inner_radius), int(max_radius) + 1):
                alpha = 255 - int(255 * (i - inner_radius) / (max_radius - inner_radius))
                draw.ellipse(
                    [(width//2 - i, height//2 - i), (width//2 + i, height//2 + i)],
                    fill=alpha
                )
            
            # Create vignette overlay
            vignette_overlay = Image.new('RGB', (width, height), vignette_color[:3])
            
            # Apply vignette
            rgb_image = Image.composite(rgb_image, vignette_overlay, mask)
        
        # Merge back with alpha channel
        r, g, b = rgb_image.split()
        return Image.merge('RGBA', (r, g, b, a))
    
    def _apply_luminance(self, image, luminance_image):
        """Apply luminance from one image while preserving color from another."""
        if OPENCV_AVAILABLE:
            # Convert to OpenCV format
            img_array = np.array(image)
            lum_array = np.array(luminance_image)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Replace L channel with luminance image
            lab[:, :, 0] = lum_array
            
            # Convert back to RGB
            result_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to PIL with original alpha
            r, g, b, a = image.split()
            result = Image.fromarray(result_array)
            r2, g2, b2 = result.split()
            return Image.merge('RGBA', (r2, g2, b2, a))
        else:
            # Simplified version using PIL
            # This is an approximation as PIL doesn't have LAB color space
            r, g, b, a = image.split()
            gray = luminance_image.convert('L')
            
            # Adjust brightness of each channel based on luminance
            r = ImageChops.multiply(r, gray)
            g = ImageChops.multiply(g, gray)
            b = ImageChops.multiply(b, gray)
            
            return Image.merge('RGBA', (r, g, b, a))
    
    def _parse_color(self, color):
        """
        Parse a color string into RGBA tuple.
        
        Args:
            color: Color string (e.g., '#RRGGBB' or '#RRGGBBAA') or tuple
            
        Returns:
            RGBA tuple (r, g, b, a)
        """
        if isinstance(color, tuple):
            # Already a tuple, ensure it has 4 components
            if len(color) == 3:
                return color + (255,)
            return color
            
        if color.startswith('#'):
            # Hex color
            if len(color) == 7:  # #RRGGBB
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                return (r, g, b, 255)
            elif len(color) == 9:  # #RRGGBBAA
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                a = int(color[7:9], 16)
                return (r, g, b, a)
        
        # Default to black
        return (0, 0, 0, 255)
    
    def add_to_history(self, image):
        """
        Add an image to the history stack for undo/redo.
        
        Args:
            image: PIL Image to add to history
        """
        # If we're not at the end of the history, truncate it
        if self.history_position < len(self.history) - 1:
            self.history = self.history[:self.history_position + 1]
        
        # Add the new image to history
        self.history.append(image.copy())
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Update position
        self.history_position = len(self.history) - 1
    
    def undo(self):
        """
        Undo the last enhancement.
        
        Returns:
            Previous image or None if no history
        """
        if not self.history or self.history_position <= 0:
            return None
        
        self.history_position -= 1
        return self.history[self.history_position].copy()
    
    def redo(self):
        """
        Redo the last undone enhancement.
        
        Returns:
            Next image or None if at end of history
        """
        if not self.history or self.history_position >= len(self.history) - 1:
            return None
        
        self.history_position += 1
        return self.history[self.history_position].copy()
    
    def reset_history(self):
        """Clear the history stack."""
        self.history = []
        self.history_position = -1
    
    def show_settings_dialog(self, parent, image=None):
        """
        Show a dialog to configure enhancement settings.
        
        Args:
            parent: Parent window
            image: Optional image for preview
            
        Returns:
            Dictionary of updated settings or None if cancelled
        """
        # Create a new toplevel window
        dialog = tk.Toplevel(parent)
        dialog.title("Image Enhancement Settings")
        dialog.geometry("800x600")
        dialog.minsize(800, 600)
        
        # Make it modal
        dialog.transient(parent)
        dialog.grab_set()
        
        # Center the window
        dialog.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for different adjustment categories
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs for each category
        basic_tab = self._create_basic_tab(notebook)
        color_tab = self._create_color_tab(notebook)
        advanced_tab = self._create_advanced_tab(notebook)
        filter_tab = self._create_filter_tab(notebook)
        effects_tab = self._create_effects_tab(notebook)
        
        # Add tabs to notebook
        notebook.add(basic_tab, text="Basic")
        notebook.add(color_tab, text="Color")
        notebook.add(advanced_tab, text="Advanced")
        notebook.add(filter_tab, text="Filters")
        notebook.add(effects_tab, text="Effects")
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Split preview into before/after
        preview_pane = ttk.PanedWindow(preview_frame, orient=tk.HORIZONTAL)
        preview_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Original image preview
        original_frame = ttk.LabelFrame(preview_pane, text="Original")
        preview_pane.add(original_frame, weight=1)
        
        original_canvas = tk.Canvas(original_frame, bg="#f0f0f0")
        original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Enhanced image preview
        enhanced_frame = ttk.LabelFrame(preview_pane, text="Enhanced")
        preview_pane.add(enhanced_frame, weight=1)
        
        enhanced_canvas = tk.Canvas(enhanced_frame, bg="#f0f0f0")
        enhanced_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Preview image references
        original_image_ref = [None]
        enhanced_image_ref = [None]
        
        # Function to update preview
        def update_preview():
            if image is None:
                return
            
            # Collect current settings
            current_settings = {
                # Basic adjustments
                'brightness': brightness_var.get(),
                'contrast': contrast_var.get(),
                'saturation': saturation_var.get(),
                'sharpness': sharpness_var.get(),
                'gamma': gamma_var.get(),
                
                # Color adjustments
                'temperature': temperature_var.get(),
                'tint': tint_var.get(),
                'vibrance': vibrance_var.get(),
                
                # Advanced adjustments
                'highlights': highlights_var.get(),
                'shadows': shadows_var.get(),
                'clarity': clarity_var.get(),
                'dehaze': dehaze_var.get(),
                
                # Filters
                'filter_type': filter_type_var.get(),
                'filter_amount': filter_amount_var.get(),
                
                # Effects
                'noise_reduction': noise_reduction_var.get(),
                'vignette': vignette_var.get(),
                'vignette_color': vignette_color_var.get(),
                
                # Auto adjustments
                'auto_contrast': auto_contrast_var.get(),
                'auto_color': auto_color_var.get(),
                'auto_tone': auto_tone_var.get()
            }
            
            try:
                # Generate preview
                preview_size = (300, 200)
                
                # Resize original for preview
                aspect_ratio = image.width / image.height
                if aspect_ratio > 1.5:
                    # Wide image
                    preview_width = 300
                    preview_height = int(preview_width / aspect_ratio)
                else:
                    # Tall or square image
                    preview_height = 200
                    preview_width = int(preview_height * aspect_ratio)
                
                # Resize original image for preview
                original_preview = image.copy()
                original_preview.thumbnail((preview_width, preview_height), Image.LANCZOS)
                
                # Apply enhancements
                enhanced_preview = self.enhance_image(original_preview, **current_settings)
                
                # Convert to PhotoImage
                original_photo = ImageTk.PhotoImage(original_preview)
                enhanced_photo = ImageTk.PhotoImage(enhanced_preview)
                
                # Update canvases
                original_canvas.delete("all")
                original_canvas.create_image(
                    original_canvas.winfo_width() // 2, 
                    original_canvas.winfo_height() // 2,
                    anchor=tk.CENTER, image=original_photo
                )
                
                enhanced_canvas.delete("all")
                enhanced_canvas.create_image(
                    enhanced_canvas.winfo_width() // 2, 
                    enhanced_canvas.winfo_height() // 2,
                    anchor=tk.CENTER, image=enhanced_photo
                )
                
                # Store references to prevent garbage collection
                original_image_ref[0] = original_photo
                enhanced_image_ref[0] = enhanced_photo
                
            except Exception as e:
                print(f"Preview error: {e}")
                original_canvas.delete("all")
                original_canvas.create_text(150, 100, text="Preview error", fill="red")
                enhanced_canvas.delete("all")
                enhanced_canvas.create_text(150, 100, text=f"Error: {str(e)}", fill="red")
        
        # Add preview button
        ttk.Button(preview_frame, text="Update Preview", command=update_preview).pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Result variable
        result = [None]
        
        # OK button
        def on_ok():
            # Collect settings
            new_settings = {
                # Basic adjustments
                'brightness': brightness_var.get(),
                'contrast': contrast_var.get(),
                'saturation': saturation_var.get(),
                'sharpness': sharpness_var.get(),
                'gamma': gamma_var.get(),
                
                # Color adjustments
                'temperature': temperature_var.get(),
                'tint': tint_var.get(),
                'vibrance': vibrance_var.get(),
                
                # Advanced adjustments
                'highlights': highlights_var.get(),
                'shadows': shadows_var.get(),
                'clarity': clarity_var.get(),
                'dehaze': dehaze_var.get(),
                
                # Filters
                'filter_type': filter_type_var.get(),
                'filter_amount': filter_amount_var.get(),
                
                # Effects
                'noise_reduction': noise_reduction_var.get(),
                'vignette': vignette_var.get(),
                'vignette_color': vignette_color_var.get(),
                
                # Auto adjustments
                'auto_contrast': auto_contrast_var.get(),
                'auto_color': auto_color_var.get(),
                'auto_tone': auto_tone_var.get()
            }
            
            # Update settings
            self.settings.update(new_settings)
            
            # Set result
            result[0] = new_settings
            
            # Close dialog
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Reset button
        def on_reset():
            # Reset all sliders to default values
            brightness_var.set(100)
            contrast_var.set(100)
            saturation_var.set(100)
            sharpness_var.set(100)
            gamma_var.set(100)
            
            temperature_var.set(0)
            tint_var.set(0)
            vibrance_var.set(0)
            
            highlights_var.set(0)
            shadows_var.set(0)
            clarity_var.set(0)
            dehaze_var.set(0)
            
            filter_type_var.set('none')
            filter_amount_var.set(100)
            
            noise_reduction_var.set(0)
            vignette_var.set(0)
            
            auto_contrast_var.set(False)
            auto_color_var.set(False)
            auto_tone_var.set(False)
            
            # Update preview
            update_preview()
        
        ttk.Button(button_frame, text="Reset", command=on_reset).pack(side=tk.LEFT, padx=5)
        
        # Variables for settings
        brightness_var = IntVar(value=self.settings.get('brightness', 100))
        contrast_var = IntVar(value=self.settings.get('contrast', 100))
        saturation_var = IntVar(value=self.settings.get('saturation', 100))
        sharpness_var = IntVar(value=self.settings.get('sharpness', 100))
        gamma_var = IntVar(value=self.settings.get('gamma', 100))
        
        temperature_var = IntVar(value=self.settings.get('temperature', 0))
        tint_var = IntVar(value=self.settings.get('tint', 0))
        vibrance_var = IntVar(value=self.settings.get('vibrance', 0))
        
        highlights_var = IntVar(value=self.settings.get('highlights', 0))
        shadows_var = IntVar(value=self.settings.get('shadows', 0))
        clarity_var = IntVar(value=self.settings.get('clarity', 0))
        dehaze_var = IntVar(value=self.settings.get('dehaze', 0))
        
        filter_type_var = StringVar(value=self.settings.get('filter_type', 'none'))
        filter_amount_var = IntVar(value=self.settings.get('filter_amount', 100))
        
        noise_reduction_var = IntVar(value=self.settings.get('noise_reduction', 0))
        vignette_var = IntVar(value=self.settings.get('vignette', 0))
        vignette_color_var = StringVar(value=self.settings.get('vignette_color', '#000000'))
        
        auto_contrast_var = BooleanVar(value=self.settings.get('auto_contrast', False))
        auto_color_var = BooleanVar(value=self.settings.get('auto_color', False))
        auto_tone_var = BooleanVar(value=self.settings.get('auto_tone', False))
        
        # Update preview when dialog is shown
        dialog.after(100, update_preview)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result[0]
    
    def _create_basic_tab(self, parent):
        """Create the basic adjustments tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Auto adjustments
        auto_frame = ttk.LabelFrame(tab, text="Auto Adjustments")
        auto_frame.pack(fill=tk.X, pady=(0, 10))
        
        auto_contrast_var = BooleanVar(value=self.settings.get('auto_contrast', False))
        ttk.Checkbutton(auto_frame, text="Auto Contrast", variable=auto_contrast_var).pack(
            anchor=tk.W, padx=5, pady=2)
        
        auto_color_var = BooleanVar(value=self.settings.get('auto_color', False))
        ttk.Checkbutton(auto_frame, text="Auto Color", variable=auto_color_var).pack(
            anchor=tk.W, padx=5, pady=2)
        
        auto_tone_var = BooleanVar(value=self.settings.get('auto_tone', False))
        ttk.Checkbutton(auto_frame, text="Auto Tone", variable=auto_tone_var).pack(
            anchor=tk.W, padx=5, pady=2)
        
        # Brightness
        brightness_var = IntVar(value=self.settings.get('brightness', 100))
        self._create_slider(tab, "Brightness:", brightness_var, 0, 200, 100)
        
        # Contrast
        contrast_var = IntVar(value=self.settings.get('contrast', 100))
        self._create_slider(tab, "Contrast:", contrast_var, 0, 200, 100)
        
        # Saturation
        saturation_var = IntVar(value=self.settings.get('saturation', 100))
        self._create_slider(tab, "Saturation:", saturation_var, 0, 200, 100)
        
        # Sharpness
        sharpness_var = IntVar(value=self.settings.get('sharpness', 100))
        self._create_slider(tab, "Sharpness:", sharpness_var, 0, 200, 100)
        
        # Gamma
        gamma_var = IntVar(value=self.settings.get('gamma', 100))
        self._create_slider(tab, "Gamma:", gamma_var, 50, 150, 100)
        
        return tab
    
    def _create_color_tab(self, parent):
        """Create the color adjustments tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Temperature
        temperature_var = IntVar(value=self.settings.get('temperature', 0))
        self._create_slider(tab, "Temperature:", temperature_var, -100, 100, 0, 
                           left_label="Cool", right_label="Warm")
        
        # Tint
        tint_var = IntVar(value=self.settings.get('tint', 0))
        self._create_slider(tab, "Tint:", tint_var, -100, 100, 0,
                           left_label="Magenta", right_label="Green")
        
        # Vibrance
        vibrance_var = IntVar(value=self.settings.get('vibrance', 0))
        self._create_slider(tab, "Vibrance:", vibrance_var, 0, 100, 0)
        
        return tab
    
    def _create_advanced_tab(self, parent):
        """Create the advanced adjustments tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Highlights
        highlights_var = IntVar(value=self.settings.get('highlights', 0))
        self._create_slider(tab, "Highlights:", highlights_var, -100, 100, 0)
        
        # Shadows
        shadows_var = IntVar(value=self.settings.get('shadows', 0))
        self._create_slider(tab, "Shadows:", shadows_var, -100, 100, 0)
        
        # Clarity
        clarity_var = IntVar(value=self.settings.get('clarity', 0))
        self._create_slider(tab, "Clarity:", clarity_var, -100, 100, 0)
        
        # Dehaze
        dehaze_var = IntVar(value=self.settings.get('dehaze', 0))
        self._create_slider(tab, "Dehaze:", dehaze_var, 0, 100, 0)
        
        return tab
    
    def _create_filter_tab(self, parent):
        """Create the filters tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Filter type
        ttk.Label(tab, text="Filter Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        filter_type_var = StringVar(value=self.settings.get('filter_type', 'none'))
        filter_combo = ttk.Combobox(tab, textvariable=filter_type_var, width=15)
        filter_combo['values'] = ('none', 'grayscale', 'sepia', 'negative', 'vintage', 'cool', 'warm', 'dramatic')
        filter_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        filter_combo.current(0)
        
        # Filter amount
        filter_amount_var = IntVar(value=self.settings.get('filter_amount', 100))
        self._create_slider(tab, "Filter Amount:", filter_amount_var, 0, 100, 100, row_offset=1)
        
        # Filter descriptions
        desc_frame = ttk.LabelFrame(tab, text="Filter Description")
        desc_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=10)
        
        filter_desc = {
            'none': "No filter applied.",
            'grayscale': "Converts the image to black and white.",
            'sepia': "Applies a warm, brownish tone for a vintage look.",
            'negative': "Inverts all colors in the image.",
            'vintage': "Creates an old photo effect with warm tones and vignette.",
            'cool': "Adds a cool, bluish tone to the image.",
            'warm': "Adds a warm, orange/yellow tone to the image.",
            'dramatic': "High contrast with dark shadows for a dramatic effect."
        }
        
        desc_label = ttk.Label(desc_frame, text=filter_desc['none'], wraplength=350)
        desc_label.pack(padx=10, pady=10)
        
        # Update description when filter type changes
        def update_desc(*args):
            desc_label.config(text=filter_desc.get(filter_type_var.get(), ""))
        
        filter_type_var.trace_add("write", update_desc)
        
        # Filter preview grid
        preview_frame = ttk.LabelFrame(tab, text="Filter Previews")
        preview_frame.grid(row=4, column=0, columnspan=3, sticky=tk.NSEW, padx=5, pady=10)
        
        # Make the preview frame expandable
        tab.grid_rowconfigure(4, weight=1)
        tab.grid_columnconfigure(2, weight=1)
        
        # We would add filter previews here in a real implementation
        ttk.Label(preview_frame, text="Filter previews would be shown here").pack(padx=10, pady=10)
        
        return tab
    
    def _create_effects_tab(self, parent):
        """Create the effects tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Noise reduction
        noise_reduction_var = IntVar(value=self.settings.get('noise_reduction', 0))
        self._create_slider(tab, "Noise Reduction:", noise_reduction_var, 0, 100, 0)
        
        # Vignette
        vignette_var = IntVar(value=self.settings.get('vignette', 0))
        self._create_slider(tab, "Vignette:", vignette_var, 0, 100, 0)
        
        # Vignette color
        ttk.Label(tab, text="Vignette Color:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        vignette_color_var = StringVar(value=self.settings.get('vignette_color', '#000000'))
        vignette_color_preview = tk.Canvas(tab, width=30, height=20, bg=vignette_color_var.get())
        vignette_color_preview.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color picker button
        def pick_vignette_color():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose vignette color", 
                                         initialcolor=vignette_color_var.get())
            if color[1]:
                vignette_color_var.set(color[1])
                vignette_color_preview.config(bg=color[1])
        
        ttk.Button(tab, text="Pick Color", command=pick_vignette_color).grid(
            row=2, column=2, padx=5, pady=5)
        
        return tab
    
    def _create_slider(self, parent, label_text, variable, min_val, max_val, default_val, 
                      row_offset=None, left_label=None, right_label=None):
        """Helper method to create a labeled slider with value display."""
        if row_offset is None:
            # Find the next available row
            row_offset = 0
            for child in parent.winfo_children():
                info = child.grid_info()
                if info:  # Only consider gridded widgets
                    row_offset = max(row_offset, int(info.get('row', 0)) + 1)
        
        # Label
        ttk.Label(parent, text=label_text).grid(
            row=row_offset, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Frame for slider and value
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row_offset, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # Optional left label
        if left_label:
            ttk.Label(slider_frame, text=left_label, font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 5))
        
        # Slider
        slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, variable=variable, 
                          orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Optional right label
        if right_label:
            ttk.Label(slider_frame, text=right_label, font=("Arial", 8)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Value display
        value_label = ttk.Label(slider_frame, width=4)
        value_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Update value label when slider changes
        def update_value(*args):
            value = variable.get()
            if default_val != 0 and default_val != 100:
                # Show as percentage of default
                percentage = int((value / default_val) * 100)
                value_label.config(text=f"{percentage}%")
            elif min_val < 0:
                # Show with sign for values that can be negative
                value_label.config(text=f"{value:+d}")
            else:
                # Show as is
                value_label.config(text=str(value))
        
        variable.trace_add("write", update_value)
        update_value()  # Initialize with current value
        
        # Reset button
        def reset_slider():
            variable.set(default_val)
        
        reset_btn = ttk.Button(slider_frame, text="R", width=2, command=reset_slider)
        reset_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        return slider
    
    def create_enhancer_panel(self, parent):
        """
        Create a panel with enhancement controls for embedding in the main application.
        
        Args:
            parent: Parent widget
            
        Returns:
            Frame containing enhancement controls
        """
        panel = ttk.LabelFrame(parent, text="Image Enhancement")
        
        # Create a notebook for different adjustment categories
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create simplified tabs for each category
        basic_tab = self._create_basic_panel_tab(notebook)
        color_tab = self._create_color_panel_tab(notebook)
        effects_tab = self._create_effects_panel_tab(notebook)
        
        # Add tabs to notebook
        notebook.add(basic_tab, text="Basic")
        notebook.add(color_tab, text="Color")
        notebook.add(effects_tab, text="Effects")
        
        # Apply button
        def on_apply():
            if hasattr(self.controller, 'apply_enhancements'):
                self.controller.apply_enhancements()
        
        ttk.Button(panel, text="Apply Enhancements", command=on_apply).pack(
            fill=tk.X, padx=5, pady=5)
        
        # Reset button
        def on_reset():
            # Reset all sliders to default values
            for var_name in ['brightness_var', 'contrast_var', 'saturation_var', 'sharpness_var',
                            'temperature_var', 'tint_var', 'vibrance_var',
                            'filter_type_var', 'filter_amount_var']:
                if hasattr(self, var_name):
                    var = getattr(self, var_name)
                    if var_name == 'filter_type_var':
                        var.set('none')
                    elif var_name.endswith('_var'):
                        if 'amount' in var_name:
                            var.set(100)
                        else:
                            var.set(0)
        
        ttk.Button(panel, text="Reset All", command=on_reset).pack(
            fill=tk.X, padx=5, pady=5)
        
        return panel
    
    def _create_basic_panel_tab(self, parent):
        """Create a simplified basic adjustments tab for the panel."""
        tab = ttk.Frame(parent, padding=5)
        
        # Store variables as attributes so they can be accessed by the controller
        self.brightness_var = IntVar(value=self.settings.get('brightness', 100))
        self._create_panel_slider(tab, "Brightness:", self.brightness_var, 0, 200, 100, 0)
        
        self.contrast_var = IntVar(value=self.settings.get('contrast', 100))
        self._create_panel_slider(tab, "Contrast:", self.contrast_var, 0, 200, 100, 1)
        
        self.saturation_var = IntVar(value=self.settings.get('saturation', 100))
        self._create_panel_slider(tab, "Saturation:", self.saturation_var, 0, 200, 100, 2)
        
        self.sharpness_var = IntVar(value=self.settings.get('sharpness', 100))
        self._create_panel_slider(tab, "Sharpness:", self.sharpness_var, 0, 200, 100, 3)
        
        return tab
    
    def _create_color_panel_tab(self, parent):
        """Create a simplified color adjustments tab for the panel."""
        tab = ttk.Frame(parent, padding=5)
        
        self.temperature_var = IntVar(value=self.settings.get('temperature', 0))
        self._create_panel_slider(tab, "Temperature:", self.temperature_var, -100, 100, 0, 0)
        
        self.tint_var = IntVar(value=self.settings.get('tint', 0))
        self._create_panel_slider(tab, "Tint:", self.tint_var, -100, 100, 0, 1)
        
        self.vibrance_var = IntVar(value=self.settings.get('vibrance', 0))
        self._create_panel_slider(tab, "Vibrance:", self.vibrance_var, 0, 100, 0, 2)
        
        # Filter type
        ttk.Label(tab, text="Filter:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        
        self.filter_type_var = StringVar(value=self.settings.get('filter_type', 'none'))
        filter_combo = ttk.Combobox(tab, textvariable=self.filter_type_var, width=10)
        filter_combo['values'] = ('none', 'grayscale', 'sepia', 'vintage', 'cool', 'warm')
        filter_combo.grid(row=3, column=1, sticky=tk.EW, padx=2, pady=2)
        
        self.filter_amount_var = IntVar(value=self.settings.get('filter_amount', 100))
        self._create_panel_slider(tab, "Amount:", self.filter_amount_var, 0, 100, 100, 4)
        
        return tab
    
    def _create_effects_panel_tab(self, parent):
        """Create a simplified effects tab for the panel."""
        tab = ttk.Frame(parent, padding=5)
        
        self.clarity_var = IntVar(value=self.settings.get('clarity', 0))
        self._create_panel_slider(tab, "Clarity:", self.clarity_var, -100, 100, 0, 0)
        
        self.dehaze_var = IntVar(value=self.settings.get('dehaze', 0))
        self._create_panel_slider(tab, "Dehaze:", self.dehaze_var, 0, 100, 0, 1)
        
        self.noise_reduction_var = IntVar(value=self.settings.get('noise_reduction', 0))
        self._create_panel_slider(tab, "Noise Reduction:", self.noise_reduction_var, 0, 100, 0, 2)
        
        self.vignette_var = IntVar(value=self.settings.get('vignette', 0))
        self._create_panel_slider(tab, "Vignette:", self.vignette_var, 0, 100, 0, 3)
        
        return tab
    
    def _create_panel_slider(self, parent, label_text, variable, min_val, max_val, default_val, row):
        """Create a compact slider for the panel."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=variable, 
                          orient=tk.HORIZONTAL)
        slider.grid(row=row, column=1, sticky=tk.EW, padx=2, pady=2)
        
        # Value display
        value_label = ttk.Label(parent, width=3)
        value_label.grid(row=row, column=2, padx=2, pady=2)
        
        # Update value label when slider changes
        def update_value(*args):
            value = variable.get()
            if default_val != 0 and default_val != 100:
                # Show as percentage of default
                percentage = int((value / default_val) * 100)
                value_label.config(text=f"{percentage}%")
            elif min_val < 0:
                # Show with sign for values that can be negative
                value_label.config(text=f"{value:+d}")
            else:
                # Show as is
                value_label.config(text=str(value))
        
        variable.trace_add("write", update_value)
        update_value()  # Initialize with current value
        
        return slider
    
    def get_current_settings(self):
        """
        Get the current enhancement settings from the panel controls.
        
        Returns:
            Dictionary of current settings
        """
        settings = {}
        
        # Check if panel variables exist
        for var_name, setting_name in [
            ('brightness_var', 'brightness'),
            ('contrast_var', 'contrast'),
            ('saturation_var', 'saturation'),
            ('sharpness_var', 'sharpness'),
            ('temperature_var', 'temperature'),
            ('tint_var', 'tint'),
            ('vibrance_var', 'vibrance'),
            ('clarity_var', 'clarity'),
            ('dehaze_var', 'dehaze'),
            ('noise_reduction_var', 'noise_reduction'),
            ('vignette_var', 'vignette'),
            ('filter_type_var', 'filter_type'),
            ('filter_amount_var', 'filter_amount')
        ]:
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                settings[setting_name] = var.get()
        
        return settings


def test_image_enhancer():
    """Test function for the image enhancer."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create a simple UI for testing
        root = tk.Tk()
        root.title("Image Enhancer Test")
        root.geometry("1000x700")
        
        # Create the image enhancer
        enhancer = ImageEnhancer()
        
        # Create a frame for controls
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Current image
        current_image = [None]
        
        # Open image function
        def open_image():
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if file_path:
                try:
                    img = Image.open(file_path)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    current_image[0] = img
                    update_preview(img)
                except Exception as e:
                    print(f"Error opening image: {e}")
        
        ttk.Button(control_frame, text="Open Image", command=open_image).pack(pady=10)
        
        # Show settings dialog
        def show_settings():
            if current_image[0] is None:
                return
                
            settings = enhancer.show_settings_dialog(root, current_image[0])
            if settings:
                # Apply enhancements with new settings
                enhanced = enhancer.enhance_image(current_image[0], **settings)
                update_preview(enhanced)
        
        ttk.Button(control_frame, text="Enhancement Settings", command=show_settings).pack(pady=10)
        
        # Create a preview area
        preview_frame = ttk.LabelFrame(root, text="Preview", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        preview_canvas = tk.Canvas(preview_frame, bg="#f0f0f0")
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Preview image reference
        preview_image_ref = [None]
        
        # Update preview function
        def update_preview(image):
            if image is None:
                return
                
            # Resize for preview
            preview_canvas_width = preview_canvas.winfo_width()
            preview_canvas_height = preview_canvas.winfo_height()
            
            if preview_canvas_width <= 1 or preview_canvas_height <= 1:
                # Canvas not yet properly sized, use default size
                preview_canvas_width = 800
                preview_canvas_height = 600
            
            # Calculate aspect ratio
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            
            if img_width > preview_canvas_width or img_height > preview_canvas_height:
                # Scale down to fit
                if aspect_ratio > 1:
                    # Wide image
                    new_width = preview_canvas_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    # Tall image
                    new_height = preview_canvas_height
                    new_width = int(new_height * aspect_ratio)
                
                preview_img = image.copy()
                preview_img.thumbnail((new_width, new_height), Image.LANCZOS)
            else:
                # Use original size
                preview_img = image.copy()
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(preview_img)
            
            # Update canvas
            preview_canvas.delete("all")
            preview_canvas.create_image(
                preview_canvas_width // 2, 
                preview_canvas_height // 2,
                anchor=tk.CENTER, image=photo
            )
            
            # Store reference to prevent garbage collection
            preview_image_ref[0] = photo
        
        # Handle window resize
        def on_resize(event):
            if current_image[0] is not None:
                # Schedule update to avoid multiple redraws during resize
                root.after_cancel(getattr(root, '_resize_job', 0))
                root._resize_job = root.after(100, lambda: update_preview(current_image[0]))
        
        preview_canvas.bind("<Configure>", on_resize)
        
        # Add the panel version of the enhancer
        panel = enhancer.create_enhancer_panel(control_frame)
        panel.pack(fill=tk.X, pady=10)
        
        # Apply enhancements from panel
        def apply_panel_enhancements():
            if current_image[0] is None:
                return
                
            settings = enhancer.get_current_settings()
            enhanced = enhancer.enhance_image(current_image[0], **settings)
            update_preview(enhanced)
        
        # Add apply method to enhancer for the panel to use
        enhancer.controller = type('obj', (object,), {
            'apply_enhancements': apply_panel_enhancements
        })
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_image_enhancer()