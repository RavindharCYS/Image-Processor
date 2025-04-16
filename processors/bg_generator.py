import os
import io
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageChops
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser

# Try to import OpenCV for additional features
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class BackgroundGenerator:
    """
    A class for generating various types of backgrounds for images with transparency.
    Supports solid colors, gradients, patterns, textures, and special effects.
    """
    
    def __init__(self, controller=None):
        """
        Initialize the background generator.
        
        Args:
            controller: The controller object that handles the application logic
        """
        self.controller = controller
        
        # Default settings
        self.settings = {
            'type': 'solid',  # 'solid', 'gradient', 'pattern', 'texture', 'special'
            'color': '#ffffff',
            'gradient_color1': '#ffffff',
            'gradient_color2': '#000000',
            'gradient_direction': 'horizontal',
            'pattern': 'checkerboard',
            'pattern_color1': '#ffffff',
            'pattern_color2': '#cccccc',
            'pattern_size': 20,
            'texture_path': None,
            'texture_opacity': 100,
            'texture_blur': 0,
            'special_effect': 'none',
            'noise_amount': 10,
            'blur_amount': 0
        }
        
        # Cache for generated backgrounds
        self.cache = {}
    
    def generate_background(self, size, **kwargs):
        """
        Generate a background image.
        
        Args:
            size: Tuple of (width, height)
            **kwargs: Additional settings to override defaults
            
        Returns:
            PIL Image with the generated background
        """
        # Update settings with any provided kwargs
        settings = self.settings.copy()
        settings.update(kwargs)
        
        # Create a cache key based on size and settings
        cache_key = self._create_cache_key(size, settings)
        
        # Check if we have a cached version
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # Choose background type
        bg_type = settings['type']
        
        # Generate the background
        if bg_type == 'solid':
            bg = self._generate_solid_background(size, settings)
        elif bg_type == 'gradient':
            bg = self._generate_gradient_background(size, settings)
        elif bg_type == 'pattern':
            bg = self._generate_pattern_background(size, settings)
        elif bg_type == 'texture':
            bg = self._generate_texture_background(size, settings)
        elif bg_type == 'special':
            bg = self._generate_special_background(size, settings)
        else:
            # Default to solid white
            bg = Image.new('RGBA', size, (255, 255, 255, 255))
        
        # Apply common post-processing
        bg = self._apply_post_processing(bg, settings)
        
        # Cache the result
        self.cache[cache_key] = bg.copy()
        
        return bg
    
    def _create_cache_key(self, size, settings):
        """Create a cache key based on size and relevant settings."""
        # Extract only the settings that affect the background
        relevant_settings = {k: v for k, v in settings.items() 
                            if k != 'texture_path'}  # Exclude file paths
        
        # Include size in the key
        key_dict = {'width': size[0], 'height': size[1], **relevant_settings}
        
        # For texture backgrounds, include the modification time of the texture file
        if settings['type'] == 'texture' and settings.get('texture_path'):
            try:
                mtime = os.path.getmtime(settings['texture_path'])
                key_dict['texture_mtime'] = mtime
            except (OSError, TypeError):
                pass
        
        # Convert to a string for hashing
        return str(key_dict)
    
    def _generate_solid_background(self, size, settings):
        """
        Generate a solid color background.
        
        Args:
            size: Tuple of (width, height)
            settings: Dictionary of settings
            
        Returns:
            PIL Image with solid color background
        """
        color = self._parse_color(settings['color'])
        return Image.new('RGBA', size, color)
    
    def _generate_gradient_background(self, size, settings):
        """
        Generate a gradient background.
        
        Args:
            size: Tuple of (width, height)
            settings: Dictionary of settings
            
        Returns:
            PIL Image with gradient background
        """
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        direction = settings['gradient_direction']
        color1 = self._parse_color(settings['gradient_color1'])
        color2 = self._parse_color(settings['gradient_color2'])
        
        # For better performance with large images, use numpy for pixel manipulation
        if max(width, height) > 1000 and OPENCV_AVAILABLE:
            return self._generate_gradient_opencv(size, direction, color1, color2)
        
        if direction == 'horizontal':
            for x in range(width):
                # Calculate gradient color at this position
                r = int(color1[0] + (color2[0] - color1[0]) * x / width)
                g = int(color1[1] + (color2[1] - color1[1]) * x / width)
                b = int(color1[2] + (color2[2] - color1[2]) * x / width)
                a = int(color1[3] + (color2[3] - color1[3]) * x / width)
                
                draw.line([(x, 0), (x, height)], fill=(r, g, b, a))
                
        elif direction == 'vertical':
            for y in range(height):
                # Calculate gradient color at this position
                r = int(color1[0] + (color2[0] - color1[0]) * y / height)
                g = int(color1[1] + (color2[1] - color1[1]) * y / height)
                b = int(color1[2] + (color2[2] - color1[2]) * y / height)
                a = int(color1[3] + (color2[3] - color1[3]) * y / height)
                
                draw.line([(0, y), (width, y)], fill=(r, g, b, a))
                
        elif direction == 'diagonal':
            # Create a diagonal gradient using a different approach for better performance
            # We'll create a gradient along a line from (0,0) to (width,height)
            max_distance = math.sqrt(width**2 + height**2)
            
            for y in range(height):
                for x in range(width):
                    # Calculate distance along diagonal
                    distance = math.sqrt(x**2 + y**2) / max_distance
                    
                    # Calculate gradient color at this position
                    r = int(color1[0] + (color2[0] - color1[0]) * distance)
                    g = int(color1[1] + (color2[1] - color1[1]) * distance)
                    b = int(color1[2] + (color2[2] - color1[2]) * distance)
                    a = int(color1[3] + (color2[3] - color1[3]) * distance)
                    
                    draw.point((x, y), fill=(r, g, b, a))
                    
        elif direction == 'radial':
            # Calculate maximum distance from center
            center_x, center_y = width // 2, height // 2
            max_distance = math.sqrt((width//2)**2 + (height//2)**2)
            
            for y in range(height):
                for x in range(width):
                    # Calculate distance from center
                    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2) / max_distance
                    
                    # Calculate gradient color at this position
                    r = int(color1[0] + (color2[0] - color1[0]) * distance)
                    g = int(color1[1] + (color2[1] - color1[1]) * distance)
                    b = int(color1[2] + (color2[2] - color1[2]) * distance)
                    a = int(color1[3] + (color2[3] - color1[3]) * distance)
                    
                    draw.point((x, y), fill=(r, g, b, a))
        
        return image
    
    def _generate_gradient_opencv(self, size, direction, color1, color2):
        """Generate gradient using OpenCV for better performance with large images."""
        width, height = size
        
        if direction == 'horizontal':
            # Create a 1-pixel high gradient
            gradient = np.zeros((1, width, 4), dtype=np.uint8)
            for x in range(width):
                r = int(color1[0] + (color2[0] - color1[0]) * x / width)
                g = int(color1[1] + (color2[1] - color1[1]) * x / width)
                b = int(color1[2] + (color2[2] - color1[2]) * x / width)
                a = int(color1[3] + (color2[3] - color1[3]) * x / width)
                gradient[0, x] = [b, g, r, a]  # OpenCV uses BGR
            
            # Repeat the gradient to fill the height
            gradient = cv2.resize(gradient, (width, height), interpolation=cv2.INTER_NEAREST)
            
        elif direction == 'vertical':
            # Create a 1-pixel wide gradient
            gradient = np.zeros((height, 1, 4), dtype=np.uint8)
            for y in range(height):
                r = int(color1[0] + (color2[0] - color1[0]) * y / height)
                g = int(color1[1] + (color2[1] - color1[1]) * y / height)
                b = int(color1[2] + (color2[2] - color1[2]) * y / height)
                a = int(color1[3] + (color2[3] - color1[3]) * y / height)
                gradient[y, 0] = [b, g, r, a]  # OpenCV uses BGR
            
            # Repeat the gradient to fill the width
            gradient = cv2.resize(gradient, (width, height), interpolation=cv2.INTER_NEAREST)
            
        elif direction == 'diagonal':
            # Create a diagonal gradient
            gradient = np.zeros((height, width, 4), dtype=np.uint8)
            max_distance = math.sqrt(width**2 + height**2)
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            distances = np.sqrt(x_coords**2 + y_coords**2) / max_distance
            
            # Calculate color components
            r = (color1[0] + (color2[0] - color1[0]) * distances).astype(np.uint8)
            g = (color1[1] + (color2[1] - color1[1]) * distances).astype(np.uint8)
            b = (color1[2] + (color2[2] - color1[2]) * distances).astype(np.uint8)
            a = (color1[3] + (color2[3] - color1[3]) * distances).astype(np.uint8)
            
            # Combine into gradient
            gradient[:, :, 0] = b
            gradient[:, :, 1] = g
            gradient[:, :, 2] = r
            gradient[:, :, 3] = a
            
        elif direction == 'radial':
            # Create a radial gradient
            gradient = np.zeros((height, width, 4), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            max_distance = math.sqrt((width//2)**2 + (height//2)**2)
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2) / max_distance
            
            # Calculate color components
            r = (color1[0] + (color2[0] - color1[0]) * distances).astype(np.uint8)
            g = (color1[1] + (color2[1] - color1[1]) * distances).astype(np.uint8)
            b = (color1[2] + (color2[2] - color1[2]) * distances).astype(np.uint8)
            a = (color1[3] + (color2[3] - color1[3]) * distances).astype(np.uint8)
            
            # Combine into gradient
            gradient[:, :, 0] = b
            gradient[:, :, 1] = g
            gradient[:, :, 2] = r
            gradient[:, :, 3] = a
        
        # Convert OpenCV image to PIL
        return Image.fromarray(cv2.cvtColor(gradient, cv2.COLOR_BGRA2RGBA))
    
    def _generate_pattern_background(self, size, settings):
        """
        Generate a pattern background.
        
        Args:
            size: Tuple of (width, height)
            settings: Dictionary of settings
            
        Returns:
            PIL Image with pattern background
        """
        pattern_type = settings.get('pattern', 'checkerboard')
        
        if pattern_type == 'checkerboard':
            return self._generate_checkerboard(size, settings)
        elif pattern_type == 'stripes':
            return self._generate_stripes(size, settings)
        elif pattern_type == 'dots':
            return self._generate_dots(size, settings)
        elif pattern_type == 'grid':
            return self._generate_grid(size, settings)
        elif pattern_type == 'triangles':
            return self._generate_triangles(size, settings)
        elif pattern_type == 'hexagons':
            return self._generate_hexagons(size, settings)
        else:
            # Default to checkerboard
            return self._generate_checkerboard(size, settings)
    
    def _generate_checkerboard(self, size, settings):
        """Generate a checkerboard pattern."""
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        color1 = self._parse_color(settings.get('pattern_color1', '#ffffff'))
        color2 = self._parse_color(settings.get('pattern_color2', '#cccccc'))
        
        cell_size = settings.get('pattern_size', 20)
        
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                color = color1 if ((x // cell_size) + (y // cell_size)) % 2 == 0 else color2
                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=color)
        
        return image
    
    def _generate_stripes(self, size, settings):
        """Generate a striped pattern."""
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        color1 = self._parse_color(settings.get('pattern_color1', '#ffffff'))
        color2 = self._parse_color(settings.get('pattern_color2', '#cccccc'))
        
        stripe_width = settings.get('pattern_size', 20)
        direction = settings.get('pattern_direction', 'horizontal')
        
        if direction == 'horizontal':
            for y in range(0, height, stripe_width * 2):
                draw.rectangle([0, y, width, y + stripe_width - 1], fill=color1)
                draw.rectangle([0, y + stripe_width, width, y + stripe_width * 2 - 1], fill=color2)
        else:
            for x in range(0, width, stripe_width * 2):
                draw.rectangle([x, 0, x + stripe_width - 1, height], fill=color1)
                draw.rectangle([x + stripe_width, 0, x + stripe_width * 2 - 1, height], fill=color2)
        
        return image
    
    def _generate_dots(self, size, settings):
        """Generate a dotted pattern."""
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        bg_color = self._parse_color(settings.get('pattern_color1', '#ffffff'))
        dot_color = self._parse_color(settings.get('pattern_color2', '#cccccc'))
        
        # Fill background
        draw.rectangle([0, 0, width, height], fill=bg_color)
        
        dot_size = settings.get('pattern_size', 10)
        spacing = settings.get('pattern_spacing', 30)
        
        for y in range(spacing // 2, height, spacing):
            for x in range(spacing // 2, width, spacing):
                draw.ellipse([x - dot_size // 2, y - dot_size // 2, 
                             x + dot_size // 2, y + dot_size // 2], fill=dot_color)
        
        return image
    
    def _generate_grid(self, size, settings):
        """Generate a grid pattern."""
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        bg_color = self._parse_color(settings.get('pattern_color1', '#ffffff'))
        line_color = self._parse_color(settings.get('pattern_color2', '#cccccc'))
        
        # Fill background
        draw.rectangle([0, 0, width, height], fill=bg_color)
        
        grid_size = settings.get('pattern_size', 20)
        line_width = max(1, grid_size // 10)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=line_color, width=line_width)
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
        
        return image
    
    def _generate_triangles(self, size, settings):
        """Generate a triangular pattern."""
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        color1 = self._parse_color(settings.get('pattern_color1', '#ffffff'))
        color2 = self._parse_color(settings.get('pattern_color2', '#cccccc'))
        
        triangle_size = settings.get('pattern_size', 30)
        
        # Calculate number of triangles needed
        num_x = width // triangle_size + 2
        num_y = height // triangle_size + 2
        
        for row in range(num_y):
            for col in range(num_x):
                x = col * triangle_size
                y = row * triangle_size
                
                # Alternate triangle orientation and color
                if (row + col) % 2 == 0:
                    # Pointing up
                    points = [
                        (x, y + triangle_size),
                        (x + triangle_size, y + triangle_size),
                        (x + triangle_size // 2, y)
                    ]
                    color = color1
                else:
                    # Pointing down
                    points = [
                        (x, y),
                        (x + triangle_size, y),
                        (x + triangle_size // 2, y + triangle_size)
                    ]
                    color = color2
                
                draw.polygon(points, fill=color)
        
        return image
    
    def _generate_hexagons(self, size, settings):
        """Generate a hexagonal pattern."""
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        color1 = self._parse_color(settings.get('pattern_color1', '#ffffff'))
        color2 = self._parse_color(settings.get('pattern_color2', '#cccccc'))
        
        hex_size = settings.get('pattern_size', 30)
        
        # Calculate hexagon dimensions
        hex_width = hex_size * 2
        hex_height = int(hex_size * math.sqrt(3))
        
        # Calculate offsets for tiling
        x_offset = hex_width * 3/4
        y_offset = hex_height
        
        # Calculate number of hexagons needed
        num_x = int(width / x_offset) + 2
        num_y = int(height / y_offset) + 2
        
        for row in range(num_y):
            for col in range(num_x):
                                # Calculate center of hexagon
                center_x = col * x_offset
                center_y = row * y_offset
                
                # Offset every other row
                if row % 2 == 1:
                    center_x += x_offset / 2
                
                # Calculate the six points of the hexagon
                points = []
                for i in range(6):
                    angle = 2 * math.pi / 6 * i + math.pi / 6
                    x = center_x + hex_size * math.cos(angle)
                    y = center_y + hex_size * math.sin(angle)
                    points.append((x, y))
                
                # Alternate colors
                color = color1 if (row + col) % 2 == 0 else color2
                
                draw.polygon(points, fill=color)
        
        return image
    
    def _generate_texture_background(self, size, settings):
        """
        Generate a textured background from an image.
        
        Args:
            size: Tuple of (width, height)
            settings: Dictionary of settings
            
        Returns:
            PIL Image with texture background
        """
        texture_path = settings.get('texture_path')
        
        if not texture_path or not os.path.exists(texture_path):
            # Fall back to solid color
            return self._generate_solid_background(size, settings)
        
        try:
            # Load texture image
            texture = Image.open(texture_path).convert('RGBA')
            
            # Apply transformations
            texture = self._process_texture(texture, size, settings)
            
            return texture
            
        except Exception as e:
            print(f"Error loading texture: {e}")
            # Fall back to solid color on error
            return self._generate_solid_background(size, settings)
    
    def _process_texture(self, texture, target_size, settings):
        """Process a texture image to fit the target size with various options."""
        width, height = target_size
        
        # Resize mode
        resize_mode = settings.get('texture_resize', 'fill')
        
        if resize_mode == 'fill':
            # Resize to fill the target size while maintaining aspect ratio
            texture = self._resize_to_fill(texture, target_size)
            
            # Crop to exact size
            left = (texture.width - width) // 2
            top = (texture.height - height) // 2
            texture = texture.crop((left, top, left + width, top + height))
            
        elif resize_mode == 'fit':
            # Resize to fit within the target size while maintaining aspect ratio
            texture = self._resize_to_fit(texture, target_size)
            
            # Create a new image with the background color
            bg_color = self._parse_color(settings.get('color', '#ffffff'))
            result = Image.new('RGBA', target_size, bg_color)
            
            # Paste the texture in the center
            left = (width - texture.width) // 2
            top = (height - texture.height) // 2
            result.paste(texture, (left, top), texture)
            
            texture = result
            
        elif resize_mode == 'stretch':
            # Stretch to exactly fit the target size
            texture = texture.resize(target_size, Image.LANCZOS)
            
        elif resize_mode == 'tile':
            # Create a new image with the target size
            result = Image.new('RGBA', target_size, (0, 0, 0, 0))
            
            # Calculate how many tiles we need
            tile_width, tile_height = texture.size
            
            # Resize the tile if it's too large
            max_tile_size = min(width, height) // 2
            if tile_width > max_tile_size or tile_height > max_tile_size:
                scale = max_tile_size / max(tile_width, tile_height)
                tile_width = int(tile_width * scale)
                tile_height = int(tile_height * scale)
                texture = texture.resize((tile_width, tile_height), Image.LANCZOS)
            
            # Tile the texture
            for y in range(0, height, tile_height):
                for x in range(0, width, tile_width):
                    result.paste(texture, (x, y), texture)
            
            texture = result
        
        # Apply opacity
        opacity = settings.get('texture_opacity', 100)
        if opacity < 100:
            # Create a new image with reduced alpha
            r, g, b, a = texture.split()
            a = a.point(lambda i: i * opacity / 100)
            texture = Image.merge('RGBA', (r, g, b, a))
        
        # Apply blur
        blur_amount = settings.get('texture_blur', 0)
        if blur_amount > 0:
            texture = texture.filter(ImageFilter.GaussianBlur(radius=blur_amount / 2))
        
        return texture
    
    def _resize_to_fill(self, image, target_size):
        """Resize image to fill the target size while maintaining aspect ratio."""
        width, height = image.size
        target_width, target_height = target_size
        
        # Calculate aspect ratios
        aspect = width / height
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Image is wider than target, scale to match height
            new_height = target_height
            new_width = int(new_height * aspect)
        else:
            # Image is taller than target, scale to match width
            new_width = target_width
            new_height = int(new_width / aspect)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def _resize_to_fit(self, image, target_size):
        """Resize image to fit within the target size while maintaining aspect ratio."""
        width, height = image.size
        target_width, target_height = target_size
        
        # Calculate aspect ratios
        aspect = width / height
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Image is wider than target, scale to match width
            new_width = target_width
            new_height = int(new_width / aspect)
        else:
            # Image is taller than target, scale to match height
            new_height = target_height
            new_width = int(new_height * aspect)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def _generate_special_background(self, size, settings):
        """
        Generate a special effect background.
        
        Args:
            size: Tuple of (width, height)
            settings: Dictionary of settings
            
        Returns:
            PIL Image with special effect background
        """
        effect = settings.get('special_effect', 'none')
        
        if effect == 'noise':
            return self._generate_noise_background(size, settings)
        elif effect == 'clouds':
            return self._generate_cloud_background(size, settings)
        elif effect == 'marble':
            return self._generate_marble_background(size, settings)
        elif effect == 'plasma':
            return self._generate_plasma_background(size, settings)
        elif effect == 'starburst':
            return self._generate_starburst_background(size, settings)
        else:
            # Default to solid color
            return self._generate_solid_background(size, settings)
    
    def _generate_noise_background(self, size, settings):
        """Generate a noise background."""
        width, height = size
        
        # Create a base color
        base_color = self._parse_color(settings.get('color', '#ffffff'))
        image = Image.new('RGBA', size, base_color)
        
        # Get noise amount
        noise_amount = settings.get('noise_amount', 10)
        
        if OPENCV_AVAILABLE:
            # Use OpenCV for faster noise generation
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            
            # Generate noise
            noise = np.random.randint(-noise_amount, noise_amount + 1, img_array.shape[:2])
            
            # Apply noise to each channel except alpha
            for i in range(3):  # RGB channels
                img_array[:, :, i] = np.clip(img_array[:, :, i] + noise, 0, 255)
            
            # Convert back to PIL
            return Image.fromarray(img_array)
        else:
            # Use PIL for noise generation
            pixels = image.load()
            
            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    
                    # Add random noise to each channel
                    noise = random.randint(-noise_amount, noise_amount)
                    r = max(0, min(255, r + noise))
                    g = max(0, min(255, g + noise))
                    b = max(0, min(255, b + noise))
                    
                    pixels[x, y] = (r, g, b, a)
            
            return image
    
    def _generate_cloud_background(self, size, settings):
        """Generate a cloud-like background using Perlin noise."""
        width, height = size
        
        # Create a base color
        base_color = self._parse_color(settings.get('color', '#ffffff'))
        cloud_color = self._parse_color(settings.get('cloud_color', '#e0e0ff'))
        
        # Create base image
        image = Image.new('RGBA', size, base_color)
        
        if OPENCV_AVAILABLE:
            # Use OpenCV for faster cloud generation
            # Generate Perlin-like noise using multiple octaves of simplex noise
            scale = settings.get('cloud_scale', 100) / 100.0
            
            # Create a noise image
            noise = np.zeros((height, width), dtype=np.float32)
            
            # Generate multiple octaves of noise
            octaves = 4
            persistence = 0.5
            amplitude = 1.0
            
            for octave in range(octaves):
                frequency = 2 ** octave
                octave_amplitude = amplitude * persistence ** octave
                
                # Generate random noise
                octave_noise = np.random.rand(height // frequency + 1, width // frequency + 1).astype(np.float32)
                
                # Resize to full size with bilinear interpolation
                octave_noise = cv2.resize(octave_noise, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # Add to total noise
                noise += octave_noise * octave_amplitude
            
            # Normalize noise to 0-1 range
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            # Apply cloud effect
            img_array = np.array(image)
            
            # Interpolate between base color and cloud color based on noise
            for i in range(3):  # RGB channels
                img_array[:, :, i] = (1 - noise) * base_color[i] + noise * cloud_color[i]
            
            # Convert back to PIL
            return Image.fromarray(img_array)
        else:
            # Simplified cloud effect using PIL
            draw = ImageDraw.Draw(image)
            
            # Draw random cloud-like shapes
            num_clouds = 20
            for _ in range(num_clouds):
                x = random.randint(0, width)
                y = random.randint(0, height)
                size = random.randint(50, 200)
                
                # Draw a soft ellipse
                for i in range(5):  # Multiple layers for softness
                    ellipse_size = size - i * 10
                    if ellipse_size <= 0:
                        continue
                    
                    opacity = int(200 * (1 - i / 5))
                    color = (*cloud_color[:3], opacity)
                    
                    draw.ellipse([x - ellipse_size, y - ellipse_size // 2, 
                                 x + ellipse_size, y + ellipse_size // 2], 
                                fill=color)
            
            # Apply Gaussian blur for softness
            image = image.filter(ImageFilter.GaussianBlur(radius=10))
            
            return image
    
    def _generate_marble_background(self, size, settings):
        """Generate a marble-like background."""
        width, height = size
        
        # Create a base color
        base_color = self._parse_color(settings.get('color', '#ffffff'))
        vein_color = self._parse_color(settings.get('vein_color', '#cccccc'))
        
        # Create base image
        image = Image.new('RGBA', size, base_color)
        
        if OPENCV_AVAILABLE:
            # Use OpenCV for faster marble generation
            # Generate Perlin-like noise
            scale = settings.get('marble_scale', 50) / 100.0
            
            # Create a noise image
            noise = np.zeros((height, width), dtype=np.float32)
            
            # Generate multiple octaves of noise
            octaves = 6
            persistence = 0.6
            
            for octave in range(octaves):
                frequency = 2 ** octave
                amplitude = persistence ** octave
                
                # Generate random noise
                octave_noise = np.random.rand(height // frequency + 1, width // frequency + 1).astype(np.float32)
                
                # Resize to full size with bilinear interpolation
                octave_noise = cv2.resize(octave_noise, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # Add to total noise
                noise += octave_noise * amplitude
            
            # Normalize noise to 0-1 range
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            # Apply sine wave distortion for marble effect
            x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
            marble = (noise + scale * np.sin(x_grid / 16.0 + noise * 4)) % 1.0
            
            # Apply marble effect
            img_array = np.array(image)
            
            # Interpolate between base color and vein color based on marble
            for i in range(3):  # RGB channels
                img_array[:, :, i] = (1 - marble) * base_color[i] + marble * vein_color[i]
            
            # Convert back to PIL
            return Image.fromarray(img_array)
        else:
            # Simplified marble effect using PIL
            # Start with a noise background
            image = self._generate_noise_background(size, {
                'color': base_color,
                'noise_amount': 5
            })
            
            # Add some random curved lines for veins
            draw = ImageDraw.Draw(image)
            
            # Draw random veins
            num_veins = 10
            for _ in range(num_veins):
                # Create a series of points for a curved line
                points = []
                x = random.randint(-width//2, width//2)
                y = random.randint(0, height)
                
                for i in range(10):
                    x += random.randint(10, 30)
                    y += random.randint(-20, 20)
                    points.append((x, y))
                
                # Draw the vein
                draw.line(points, fill=vein_color, width=random.randint(1, 3))
            
            # Apply Gaussian blur for softness
            image = image.filter(ImageFilter.GaussianBlur(radius=3))
            
            return image
    
    def _generate_plasma_background(self, size, settings):
        """Generate a plasma-like background."""
        width, height = size
        
        # Get colors
        color1 = self._parse_color(settings.get('plasma_color1', '#ff0000'))
        color2 = self._parse_color(settings.get('plasma_color2', '#0000ff'))
        
        # Create base image
        image = Image.new('RGBA', size, (0, 0, 0, 255))
        
        if OPENCV_AVAILABLE:
            # Use OpenCV for faster plasma generation
            # Generate plasma using a combination of sine waves
            scale = settings.get('plasma_scale', 50) / 50.0
            
            # Create coordinate grids
            x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
            
            # Generate plasma pattern
            plasma = np.sin(x_grid / (16.0 * scale)) + np.sin(y_grid / (8.0 * scale)) + \
                    np.sin((x_grid + y_grid) / (16.0 * scale)) + \
                    np.sin(np.sqrt(((x_grid - width/2)**2 + (y_grid - height/2)**2)) / (8.0 * scale))
            
            # Normalize to 0-1 range
            plasma = (plasma - plasma.min()) / (plasma.max() - plasma.min())
            
            # Create image array
            img_array = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Interpolate between colors based on plasma value
            for i in range(3):  # RGB channels
                img_array[:, :, i] = (1 - plasma) * color1[i] + plasma * color2[i]
            
            # Set alpha channel
            img_array[:, :, 3] = 255
            
            # Convert to PIL image
            return Image.fromarray(img_array)
        else:
            # Simplified plasma effect using PIL
            pixels = image.load()
            
            for y in range(height):
                for x in range(width):
                    # Generate plasma value using sine waves
                    value = (
                        math.sin(x / 16.0) + 
                        math.sin(y / 8.0) + 
                        math.sin((x + y) / 16.0) + 
                        math.sin(math.sqrt(((x - width/2)**2 + (y - height/2)**2)) / 8.0)
                    ) / 4.0
                    
                    # Normalize to 0-1 range
                    value = (value + 1) / 2
                    
                    # Interpolate between colors
                    r = int((1 - value) * color1[0] + value * color2[0])
                    g = int((1 - value) * color1[1] + value * color2[1])
                    b = int((1 - value) * color1[2] + value * color2[2])
                    
                    pixels[x, y] = (r, g, b, 255)
            
            return image
    
    def _generate_starburst_background(self, size, settings):
        """Generate a starburst/radial ray background."""
        width, height = size
        
        # Get colors
        color1 = self._parse_color(settings.get('starburst_color1', '#ffffff'))
        color2 = self._parse_color(settings.get('starburst_color2', '#ffff00'))
        
        # Create base image
        image = Image.new('RGBA', size, (0, 0, 0, 255))
        
        # Calculate center
        center_x = width // 2
        center_y = height // 2
        
        # Get number of rays
        num_rays = settings.get('starburst_rays', 12)
        
        if OPENCV_AVAILABLE:
            # Use OpenCV for faster starburst generation
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            # Calculate angles from center
            x_diff = x_coords - center_x
            y_diff = y_coords - center_y
            angles = np.arctan2(y_diff, x_diff)
            
            # Convert to 0-1 range
            angles = (angles + np.pi) / (2 * np.pi)
            
            # Create ray pattern
            rays = (np.cos(angles * num_rays * np.pi) + 1) / 2
            
            # Create image array
            img_array = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Interpolate between colors based on ray value
            for i in range(3):  # RGB channels
                img_array[:, :, i] = (1 - rays) * color1[i] + rays * color2[i]
            
            # Set alpha channel
            img_array[:, :, 3] = 255
            
            # Convert to PIL image
            return Image.fromarray(img_array)
        else:
            # Use PIL for starburst generation
            draw = ImageDraw.Draw(image)
            
                        # Calculate maximum radius
            max_radius = math.sqrt(width**2 + height**2) / 2
            
            # Draw alternating colored wedges
            for i in range(num_rays):
                start_angle = i * 360 / num_rays
                end_angle = (i + 1) * 360 / num_rays
                
                # Choose color based on whether ray index is even or odd
                color = color1 if i % 2 == 0 else color2
                
                # Draw a wedge/pie slice
                draw.pieslice([center_x - max_radius, center_y - max_radius,
                              center_x + max_radius, center_y + max_radius],
                             start_angle, end_angle, fill=color)
            
            return image
    
    def _apply_post_processing(self, image, settings):
        """
        Apply post-processing effects to the background.
        
        Args:
            image: PIL Image to process
            settings: Dictionary of settings
            
        Returns:
            Processed PIL Image
        """
        # Apply noise if requested
        noise_amount = settings.get('noise_amount', 0)
        if noise_amount > 0:
            image = self._add_noise(image, noise_amount)
        
        # Apply blur if requested
        blur_amount = settings.get('blur_amount', 0)
        if blur_amount > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_amount / 2))
        
        # Apply brightness/contrast adjustments if requested
        brightness = settings.get('brightness', 100)
        if brightness != 100:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness / 100)
        
        contrast = settings.get('contrast', 100)
        if contrast != 100:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast / 100)
        
        return image
    
    def _add_noise(self, image, amount):
        """Add noise to an image."""
        if OPENCV_AVAILABLE:
            # Use OpenCV for faster noise addition
            img_array = np.array(image)
            
            # Generate noise
            noise = np.random.randint(-amount, amount + 1, img_array.shape[:3])
            
            # Apply noise to each channel except alpha
            for i in range(3):  # RGB channels
                img_array[:, :, i] = np.clip(img_array[:, :, i] + noise[:, :, i], 0, 255)
            
            # Convert back to PIL
            return Image.fromarray(img_array)
        else:
            # Use PIL for noise addition
            pixels = image.load()
            width, height = image.size
            
            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    
                    # Add random noise to each channel
                    r = max(0, min(255, r + random.randint(-amount, amount)))
                    g = max(0, min(255, g + random.randint(-amount, amount)))
                    b = max(0, min(255, b + random.randint(-amount, amount)))
                    
                    pixels[x, y] = (r, g, b, a)
            
            return image
    
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
        
        # Default to white
        return (255, 255, 255, 255)
    
    def clear_cache(self):
        """Clear the background cache."""
        self.cache = {}
    
    def show_settings_dialog(self, parent):
        """
        Show a dialog to configure background generation settings.
        
        Args:
            parent: Parent window
            
        Returns:
            Dictionary of updated settings or None if cancelled
        """
        # Create a new toplevel window
        dialog = tk.Toplevel(parent)
        dialog.title("Background Settings")
        dialog.geometry("500x600")
        dialog.resizable(False, False)
        
        # Make it modal
        dialog.transient(parent)
        dialog.grab_set()
        
        # Center the window
        dialog.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Create a frame with padding
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for different background types
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Background type variable
        bg_type_var = tk.StringVar(value=self.settings['type'])
        
        # Create tabs for each background type
        solid_tab = self._create_solid_tab(notebook, bg_type_var)
        gradient_tab = self._create_gradient_tab(notebook, bg_type_var)
        pattern_tab = self._create_pattern_tab(notebook, bg_type_var)
        texture_tab = self._create_texture_tab(notebook, bg_type_var)
        special_tab = self._create_special_tab(notebook, bg_type_var)
        
        # Add tabs to notebook
        notebook.add(solid_tab, text="Solid")
        notebook.add(gradient_tab, text="Gradient")
        notebook.add(pattern_tab, text="Pattern")
        notebook.add(texture_tab, text="Texture")
        notebook.add(special_tab, text="Special")
        
        # Set the active tab based on current settings
        if bg_type_var.get() == 'solid':
            notebook.select(0)
        elif bg_type_var.get() == 'gradient':
            notebook.select(1)
        elif bg_type_var.get() == 'pattern':
            notebook.select(2)
        elif bg_type_var.get() == 'texture':
            notebook.select(3)
        elif bg_type_var.get() == 'special':
            notebook.select(4)
        
        # Update background type when tab changes
        def on_tab_change(event):
            tab_id = notebook.index(notebook.select())
            if tab_id == 0:
                bg_type_var.set('solid')
            elif tab_id == 1:
                bg_type_var.set('gradient')
            elif tab_id == 2:
                bg_type_var.set('pattern')
            elif tab_id == 3:
                bg_type_var.set('texture')
            elif tab_id == 4:
                bg_type_var.set('special')
        
        notebook.bind("<<NotebookTabChanged>>", on_tab_change)
        
        # Common settings frame
        common_frame = ttk.LabelFrame(main_frame, text="Common Settings")
        common_frame.pack(fill=tk.X, pady=10)
        
        # Noise amount
        ttk.Label(common_frame, text="Noise:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        noise_var = tk.IntVar(value=self.settings.get('noise_amount', 0))
        noise_scale = ttk.Scale(common_frame, from_=0, to=50, variable=noise_var, orient=tk.HORIZONTAL)
        noise_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        noise_label = ttk.Label(common_frame, text=str(noise_var.get()))
        noise_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_noise_label(*args):
            noise_label.config(text=str(noise_var.get()))
        
        noise_var.trace_add("write", update_noise_label)
        
        # Blur amount
        ttk.Label(common_frame, text="Blur:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        blur_var = tk.IntVar(value=self.settings.get('blur_amount', 0))
        blur_scale = ttk.Scale(common_frame, from_=0, to=20, variable=blur_var, orient=tk.HORIZONTAL)
        blur_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        blur_label = ttk.Label(common_frame, text=str(blur_var.get()))
        blur_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_blur_label(*args):
            blur_label.config(text=str(blur_var.get()))
        
        blur_var.trace_add("write", update_blur_label)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Preview canvas
        preview_canvas = tk.Canvas(preview_frame, width=200, height=150, bg="#f0f0f0")
        preview_canvas.pack(padx=10, pady=10)
        
        # Preview image reference
        preview_image_ref = [None]
        
        # Function to update preview
        def update_preview():
            # Collect current settings
            current_settings = {
                'type': bg_type_var.get(),
                'color': solid_color_var.get(),
                'gradient_color1': gradient_color1_var.get(),
                'gradient_color2': gradient_color2_var.get(),
                'gradient_direction': gradient_direction_var.get(),
                'pattern': pattern_type_var.get(),
                'pattern_color1': pattern_color1_var.get(),
                'pattern_color2': pattern_color2_var.get(),
                'pattern_size': pattern_size_var.get(),
                'texture_path': texture_path_var.get(),
                'texture_opacity': texture_opacity_var.get(),
                'texture_blur': texture_blur_var.get(),
                'texture_resize': texture_resize_var.get(),
                'special_effect': special_effect_var.get(),
                'noise_amount': noise_var.get(),
                'blur_amount': blur_var.get(),
                'plasma_color1': plasma_color1_var.get(),
                'plasma_color2': plasma_color2_var.get(),
                'plasma_scale': plasma_scale_var.get(),
                'starburst_color1': starburst_color1_var.get(),
                'starburst_color2': starburst_color2_var.get(),
                'starburst_rays': starburst_rays_var.get()
            }
            
            # Generate preview
            try:
                preview_size = (200, 150)
                preview_bg = self.generate_background(preview_size, **current_settings)
                
                # Convert to PhotoImage
                preview_image = ImageTk.PhotoImage(preview_bg)
                
                # Update canvas
                preview_canvas.delete("all")
                preview_canvas.create_image(0, 0, anchor=tk.NW, image=preview_image)
                
                # Store reference to prevent garbage collection
                preview_image_ref[0] = preview_image
                
            except Exception as e:
                print(f"Preview error: {e}")
                preview_canvas.delete("all")
                preview_canvas.create_text(100, 75, text="Preview error", fill="red")
        
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
                'type': bg_type_var.get(),
                'color': solid_color_var.get(),
                'gradient_color1': gradient_color1_var.get(),
                'gradient_color2': gradient_color2_var.get(),
                'gradient_direction': gradient_direction_var.get(),
                'pattern': pattern_type_var.get(),
                'pattern_color1': pattern_color1_var.get(),
                'pattern_color2': pattern_color2_var.get(),
                'pattern_size': pattern_size_var.get(),
                'texture_path': texture_path_var.get(),
                'texture_opacity': texture_opacity_var.get(),
                'texture_blur': texture_blur_var.get(),
                'texture_resize': texture_resize_var.get(),
                'special_effect': special_effect_var.get(),
                'noise_amount': noise_var.get(),
                'blur_amount': blur_var.get(),
                'plasma_color1': plasma_color1_var.get(),
                'plasma_color2': plasma_color2_var.get(),
                'plasma_scale': plasma_scale_var.get(),
                'starburst_color1': starburst_color1_var.get(),
                'starburst_color2': starburst_color2_var.get(),
                'starburst_rays': starburst_rays_var.get()
            }
            
            # Update settings
            self.settings.update(new_settings)
            
            # Clear cache
            self.clear_cache()
            
            # Set result
            result[0] = new_settings
            
            # Close dialog
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Variables for settings
        solid_color_var = tk.StringVar(value=self.settings.get('color', '#ffffff'))
        gradient_color1_var = tk.StringVar(value=self.settings.get('gradient_color1', '#ffffff'))
        gradient_color2_var = tk.StringVar(value=self.settings.get('gradient_color2', '#000000'))
        gradient_direction_var = tk.StringVar(value=self.settings.get('gradient_direction', 'horizontal'))
        pattern_type_var = tk.StringVar(value=self.settings.get('pattern', 'checkerboard'))
        pattern_color1_var = tk.StringVar(value=self.settings.get('pattern_color1', '#ffffff'))
        pattern_color2_var = tk.StringVar(value=self.settings.get('pattern_color2', '#cccccc'))
        pattern_size_var = tk.IntVar(value=self.settings.get('pattern_size', 20))
        texture_path_var = tk.StringVar(value=self.settings.get('texture_path', ''))
        texture_opacity_var = tk.IntVar(value=self.settings.get('texture_opacity', 100))
        texture_blur_var = tk.IntVar(value=self.settings.get('texture_blur', 0))
        texture_resize_var = tk.StringVar(value=self.settings.get('texture_resize', 'fill'))
        special_effect_var = tk.StringVar(value=self.settings.get('special_effect', 'none'))
        plasma_color1_var = tk.StringVar(value=self.settings.get('plasma_color1', '#ff0000'))
        plasma_color2_var = tk.StringVar(value=self.settings.get('plasma_color2', '#0000ff'))
        plasma_scale_var = tk.IntVar(value=self.settings.get('plasma_scale', 50))
        starburst_color1_var = tk.StringVar(value=self.settings.get('starburst_color1', '#ffffff'))
        starburst_color2_var = tk.StringVar(value=self.settings.get('starburst_color2', '#ffff00'))
        starburst_rays_var = tk.IntVar(value=self.settings.get('starburst_rays', 12))
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result[0]
    
    def _create_solid_tab(self, parent, bg_type_var):
        """Create the solid color tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Set background type when this tab is selected
        def on_tab_selected():
            bg_type_var.set('solid')
        
        # Color selection
        ttk.Label(tab, text="Color:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Color display
        solid_color_var = tk.StringVar(value=self.settings.get('color', '#ffffff'))
        solid_color_preview = tk.Canvas(tab, width=30, height=20, bg=solid_color_var.get())
        solid_color_preview.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color picker button
        def pick_solid_color():
            color = colorchooser.askcolor(title="Choose background color", 
                                         initialcolor=solid_color_var.get())
            if color[1]:
                solid_color_var.set(color[1])
                solid_color_preview.config(bg=color[1])
        
        ttk.Button(tab, text="Pick Color", command=pick_solid_color).grid(
            row=0, column=2, padx=5, pady=5)
        
        return tab
    
    def _create_gradient_tab(self, parent, bg_type_var):
        """Create the gradient tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Set background type when this tab is selected
        def on_tab_selected():
            bg_type_var.set('gradient')
        
        # Start color
        ttk.Label(tab, text="Start Color:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        gradient_color1_var = tk.StringVar(value=self.settings.get('gradient_color1', '#ffffff'))
        gradient_color1_preview = tk.Canvas(tab, width=30, height=20, bg=gradient_color1_var.get())
        gradient_color1_preview.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Start color picker button
        def pick_gradient_color1():
            color = colorchooser.askcolor(title="Choose start color", 
                                         initialcolor=gradient_color1_var.get())
            if color[1]:
                gradient_color1_var.set(color[1])
                gradient_color1_preview.config(bg=color[1])
        
        ttk.Button(tab, text="Pick Color", command=pick_gradient_color1).grid(
            row=0, column=2, padx=5, pady=5)
        
        # End color
        ttk.Label(tab, text="End Color:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        gradient_color2_var = tk.StringVar(value=self.settings.get('gradient_color2', '#000000'))
        gradient_color2_preview = tk.Canvas(tab, width=30, height=20, bg=gradient_color2_var.get())
        gradient_color2_preview.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # End color picker button
        def pick_gradient_color2():
            color = colorchooser.askcolor(title="Choose end color", 
                                         initialcolor=gradient_color2_var.get())
            if color[1]:
                gradient_color2_var.set(color[1])
                gradient_color2_preview.config(bg=color[1])
        
        ttk.Button(tab, text="Pick Color", command=pick_gradient_color2).grid(
            row=1, column=2, padx=5, pady=5)
        
        # Direction
        ttk.Label(tab, text="Direction:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        gradient_direction_var = tk.StringVar(value=self.settings.get('gradient_direction', 'horizontal'))
        direction_combo = ttk.Combobox(tab, textvariable=gradient_direction_var, width=15)
        direction_combo['values'] = ('horizontal', 'vertical', 'diagonal', 'radial')
        direction_combo.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        return tab
    
    def _create_pattern_tab(self, parent, bg_type_var):
        """Create the pattern tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Set background type when this tab is selected
        def on_tab_selected():
            bg_type_var.set('pattern')
        
        # Pattern type
        ttk.Label(tab, text="Pattern Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        pattern_type_var = tk.StringVar(value=self.settings.get('pattern', 'checkerboard'))
        pattern_combo = ttk.Combobox(tab, textvariable=pattern_type_var, width=15)
        pattern_combo['values'] = ('checkerboard', 'stripes', 'dots', 'grid', 'triangles', 'hexagons')
        pattern_combo.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Color 1
        ttk.Label(tab, text="Color 1:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        pattern_color1_var = tk.StringVar(value=self.settings.get('pattern_color1', '#ffffff'))
        pattern_color1_preview = tk.Canvas(tab, width=30, height=20, bg=pattern_color1_var.get())
        pattern_color1_preview.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color 1 picker button
        def pick_pattern_color1():
            color = colorchooser.askcolor(title="Choose pattern color 1", 
                                         initialcolor=pattern_color1_var.get())
            if color[1]:
                pattern_color1_var.set(color[1])
                pattern_color1_preview.config(bg=color[1])
        
        ttk.Button(tab, text="Pick Color", command=pick_pattern_color1).grid(
            row=1, column=2, padx=5, pady=5)
        
        # Color 2
        ttk.Label(tab, text="Color 2:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        pattern_color2_var = tk.StringVar(value=self.settings.get('pattern_color2', '#cccccc'))
        pattern_color2_preview = tk.Canvas(tab, width=30, height=20, bg=pattern_color2_var.get())
        pattern_color2_preview.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color 2 picker button
        def pick_pattern_color2():
            color = colorchooser.askcolor(title="Choose pattern color 2", 
                                         initialcolor=pattern_color2_var.get())
            if color[1]:
                pattern_color2_var.set(color[1])
                pattern_color2_preview.config(bg=color[1])
        
        ttk.Button(tab, text="Pick Color", command=pick_pattern_color2).grid(
            row=2, column=2, padx=5, pady=5)
        
        # Pattern size
        ttk.Label(tab, text="Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        pattern_size_var = tk.IntVar(value=self.settings.get('pattern_size', 20))
        pattern_size_scale = ttk.Scale(tab, from_=5, to=50, variable=pattern_size_var, 
                                      orient=tk.HORIZONTAL)
        pattern_size_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        pattern_size_label = ttk.Label(tab, text=str(pattern_size_var.get()))
        pattern_size_label.grid(row=3, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_pattern_size_label(*args):
            pattern_size_label.config(text=str(pattern_size_var.get()))
        
        pattern_size_var.trace_add("write", update_pattern_size_label)
        
        # Direction (for stripes)
        ttk.Label(tab, text="Direction:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        
        pattern_direction_var = tk.StringVar(value=self.settings.get('pattern_direction', 'horizontal'))
        ttk.Radiobutton(tab, text="Horizontal", variable=pattern_direction_var, 
                       value="horizontal").grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(tab, text="Vertical", variable=pattern_direction_var, 
                       value="vertical").grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        return tab
    
    def _create_texture_tab(self, parent, bg_type_var):
        """Create the texture tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Set background type when this tab is selected
        def on_tab_selected():
            bg_type_var.set('texture')
        
        # Texture image
        ttk.Label(tab, text="Texture Image:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        texture_path_var = tk.StringVar(value=self.settings.get('texture_path', ''))
        texture_path_entry = ttk.Entry(tab, textvariable=texture_path_var, width=25)
        texture_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Browse button
        def browse_texture():
            path = filedialog.askopenfilename(
                title="Select Texture Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            )
            if path:
                texture_path_var.set(path)
                # Try to load and display preview
                try:
                    img = Image.open(path)
                    img.thumbnail((100, 100))
                    photo = ImageTk.PhotoImage(img)
                    texture_preview_label.config(image=photo)
                    texture_preview_label.image = photo
                except Exception:
                    texture_preview_label.config(image=None, text="Preview not available")
        
        ttk.Button(tab, text="Browse...", command=browse_texture).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Texture preview
        texture_preview_frame = ttk.LabelFrame(tab, text="Preview")
        texture_preview_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=tk.EW)
        
        texture_preview_label = ttk.Label(texture_preview_frame, text="No texture selected")
        texture_preview_label.pack(padx=5, pady=5)
        
        # Try to load and display current texture if set
        if self.settings.get('texture_path') and os.path.exists(self.settings.get('texture_path')):
            try:
                img = Image.open(self.settings.get('texture_path'))
                img.thumbnail((100, 100))
                photo = ImageTk.PhotoImage(img)
                texture_preview_label.config(image=photo)
                texture_preview_label.image = photo
            except Exception:
                pass
        
        # Resize mode
        ttk.Label(tab, text="Resize Mode:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        texture_resize_var = tk.StringVar(value=self.settings.get('texture_resize', 'fill'))
        resize_combo = ttk.Combobox(tab, textvariable=texture_resize_var, width=15)
        resize_combo['values'] = ('fill', 'fit', 'stretch', 'tile')
        resize_combo.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Opacity
        ttk.Label(tab, text="Opacity:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        texture_opacity_var = tk.IntVar(value=self.settings.get('texture_opacity', 100))
        opacity_scale = ttk.Scale(tab, from_=0, to=100, variable=texture_opacity_var, 
                                 orient=tk.HORIZONTAL)
        opacity_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        opacity_label = ttk.Label(tab, text=f"{texture_opacity_var.get()}%")
        opacity_label.grid(row=3, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_opacity_label(*args):
            opacity_label.config(text=f"{texture_opacity_var.get()}%")
        
        texture_opacity_var.trace_add("write", update_opacity_label)
        
        # Blur
        ttk.Label(tab, text="Blur:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        
        texture_blur_var = tk.IntVar(value=self.settings.get('texture_blur', 0))
        blur_scale = ttk.Scale(tab, from_=0, to=10, variable=texture_blur_var, 
                              orient=tk.HORIZONTAL)
        blur_scale.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
        
        blur_label = ttk.Label(tab, text=str(texture_blur_var.get()))
        blur_label.grid(row=4, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_blur_label(*args):
            blur_label.config(text=str(texture_blur_var.get()))
        
        texture_blur_var.trace_add("write", update_blur_label)
        
        return tab
    
    def _create_special_tab(self, parent, bg_type_var):
        """Create the special effects tab."""
        tab = ttk.Frame(parent, padding=10)
        
        # Set background type when this tab is selected
        def on_tab_selected():
            bg_type_var.set('special')
        
        # Effect type
        ttk.Label(tab, text="Effect Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        special_effect_var = tk.StringVar(value=self.settings.get('special_effect', 'none'))
        effect_combo = ttk.Combobox(tab, textvariable=special_effect_var, width=15)
        effect_combo['values'] = ('noise', 'clouds', 'marble', 'plasma', 'starburst')
        effect_combo.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Create a notebook for different effect settings
        effect_notebook = ttk.Notebook(tab)
        effect_notebook.grid(row=1, column=0, columnspan=3, sticky=tk.NSEW, pady=10)
        
        # Noise settings
        noise_frame = ttk.Frame(effect_notebook, padding=10)
        effect_notebook.add(noise_frame, text="Noise")
        
        ttk.Label(noise_frame, text="Base Color:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        noise_color_var = tk.StringVar(value=self.settings.get('color', '#ffffff'))
        noise_color_preview = tk.Canvas(noise_frame, width=30, height=20, bg=noise_color_var.get())
        noise_color_preview.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color picker button
        def pick_noise_color():
            color = colorchooser.askcolor(title="Choose base color", 
                                         initialcolor=noise_color_var.get())
            if color[1]:
                noise_color_var.set(color[1])
                noise_color_preview.config(bg=color[1])
        
        ttk.Button(noise_frame, text="Pick Color", command=pick_noise_color).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Plasma settings
        plasma_frame = ttk.Frame(effect_notebook, padding=10)
        effect_notebook.add(plasma_frame, text="Plasma")
        
        ttk.Label(plasma_frame, text="Color 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        plasma_color1_var = tk.StringVar(value=self.settings.get('plasma_color1', '#ff0000'))
        plasma_color1_preview = tk.Canvas(plasma_frame, width=30, height=20, bg=plasma_color1_var.get())
        plasma_color1_preview.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color 1 picker button
        def pick_plasma_color1():
            color = colorchooser.askcolor(title="Choose plasma color 1", 
                                         initialcolor=plasma_color1_var.get())
            if color[1]:
                plasma_color1_var.set(color[1])
                plasma_color1_preview.config(bg=color[1])
        
        ttk.Button(plasma_frame, text="Pick Color", command=pick_plasma_color1).grid(
            row=0, column=2, padx=5, pady=5)
        
        ttk.Label(plasma_frame, text="Color 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        plasma_color2_var = tk.StringVar(value=self.settings.get('plasma_color2', '#0000ff'))
        plasma_color2_preview = tk.Canvas(plasma_frame, width=30, height=20, bg=plasma_color2_var.get())
        plasma_color2_preview.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color 2 picker button
        def pick_plasma_color2():
            color = colorchooser.askcolor(title="Choose plasma color 2", 
                                         initialcolor=plasma_color2_var.get())
            if color[1]:
                plasma_color2_var.set(color[1])
                plasma_color2_preview.config(bg=color[1])
        
        ttk.Button(plasma_frame, text="Pick Color", command=pick_plasma_color2).grid(
            row=1, column=2, padx=5, pady=5)
        
        # Scale
        ttk.Label(plasma_frame, text="Scale:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        plasma_scale_var = tk.IntVar(value=self.settings.get('plasma_scale', 50))
        plasma_scale = ttk.Scale(plasma_frame, from_=10, to=100, variable=plasma_scale_var, 
                                orient=tk.HORIZONTAL)
        plasma_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        plasma_scale_label = ttk.Label(plasma_frame, text=str(plasma_scale_var.get()))
        plasma_scale_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_plasma_scale_label(*args):
            plasma_scale_label.config(text=str(plasma_scale_var.get()))
        
        plasma_scale_var.trace_add("write", update_plasma_scale_label)
        
        # Starburst settings
        starburst_frame = ttk.Frame(effect_notebook, padding=10)
        effect_notebook.add(starburst_frame, text="Starburst")
        
        ttk.Label(starburst_frame, text="Color 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        starburst_color1_var = tk.StringVar(value=self.settings.get('starburst_color1', '#ffffff'))
        starburst_color1_preview = tk.Canvas(starburst_frame, width=30, height=20, 
                                           bg=starburst_color1_var.get())
        starburst_color1_preview.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color 1 picker button
        def pick_starburst_color1():
            color = colorchooser.askcolor(title="Choose starburst color 1", 
                                         initialcolor=starburst_color1_var.get())
            if color[1]:
                starburst_color1_var.set(color[1])
                starburst_color1_preview.config(bg=color[1])
        
        ttk.Button(starburst_frame, text="Pick Color", command=pick_starburst_color1).grid(
            row=0, column=2, padx=5, pady=5)
        
        ttk.Label(starburst_frame, text="Color 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        starburst_color2_var = tk.StringVar(value=self.settings.get('starburst_color2', '#ffff00'))
        starburst_color2_preview = tk.Canvas(starburst_frame, width=30, height=20, 
                                           bg=starburst_color2_var.get())
        starburst_color2_preview.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color 2 picker button
        def pick_starburst_color2():
            color = colorchooser.askcolor(title="Choose starburst color 2", 
                                         initialcolor=starburst_color2_var.get())
            if color[1]:
                starburst_color2_var.set(color[1])
                starburst_color2_preview.config(bg=color[1])
        
        ttk.Button(starburst_frame, text="Pick Color", command=pick_starburst_color2).grid(
            row=1, column=2, padx=5, pady=5)
        
        # Number of rays
        ttk.Label(starburst_frame, text="Rays:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        starburst_rays_var = tk.IntVar(value=self.settings.get('starburst_rays', 12))
        starburst_rays_scale = ttk.Scale(starburst_frame, from_=4, to=36, variable=starburst_rays_var, 
                                        orient=tk.HORIZONTAL)
        starburst_rays_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        starburst_rays_label = ttk.Label(starburst_frame, text=str(starburst_rays_var.get()))
        starburst_rays_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_starburst_rays_label(*args):
            starburst_rays_label.config(text=str(starburst_rays_var.get()))
        
        starburst_rays_var.trace_add("write", update_starburst_rays_label)
        
        # Show appropriate tab based on selected effect
        def update_effect_tab(*args):
            effect = special_effect_var.get()
            if effect == 'noise':
                effect_notebook.select(0)
            elif effect == 'plasma':
                effect_notebook.select(1)
            elif effect == 'starburst':
                effect_notebook.select(2)
        
        special_effect_var.trace_add("write", update_effect_tab)
        update_effect_tab()
        
        return tab


def test_background_generator():
    """Test function for the background generator."""
    try:
        import tkinter as tk
        
        # Create a simple UI for testing
        root = tk.Tk()
        root.title("Background Generator Test")
        root.geometry("800x600")
        
        # Create the background generator
        generator = BackgroundGenerator()
        
        # Create a frame for controls
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create a button to test settings dialog
        def show_settings():
            settings = generator.show_settings_dialog(root)
            if settings:
                update_preview()
        
        ttk.Button(control_frame, text="Show Settings", command=show_settings).pack(pady=10)
        
        # Create a button to generate background
        def update_preview():
            try:
                result = generator.generate_background((400, 300))
                if result:
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(result)
                    preview_label.config(image=photo)
                    preview_label.image = photo
            except Exception as e:
                print(f"Error generating background: {e}")
        
        ttk.Button(control_frame, text="Update Preview", command=update_preview).pack(pady=10)
        
        # Create a preview area
        preview_frame = ttk.LabelFrame(root, text="Preview", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        preview_label = ttk.Label(preview_frame)
        preview_label.pack(padx=10, pady=10)
        
        # Generate initial preview
        update_preview()
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_background_generator()