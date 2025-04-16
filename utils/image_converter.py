import os
import io
import math
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Try to import additional libraries for enhanced functionality
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pyheif
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

try:
    from wand.image import Image as WandImage
    WAND_AVAILABLE = True
except ImportError:
    WAND_AVAILABLE = False


class ImageConverter:
    """
    A class for converting images between different formats and color modes,
    with support for various optimization and compression options.
    """
    
    def __init__(self, controller=None):
        """
        Initialize the image converter.
        
        Args:
            controller: The controller object that handles the application logic
        """
        self.controller = controller
        
        # Default settings
        self.settings = {
            'default_format': 'PNG',
            'jpeg_quality': 90,
            'png_compression': 9,
            'webp_quality': 80,
            'webp_lossless': False,
            'tiff_compression': 'tiff_deflate',
            'gif_optimize': True,
            'preserve_metadata': True,
            'default_dpi': 72
        }
        
        # Supported formats
        self.supported_formats = {
            'input': ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF', 'WEBP'],
            'output': ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF', 'WEBP', 'PDF']
        }
        
        # Add HEIF/HEIC support if available
        if HEIF_AVAILABLE:
            self.supported_formats['input'].append('HEIF')
        
        # Add additional formats if ImageMagick is available
        if WAND_AVAILABLE:
            self.supported_formats['input'].extend(['PSD', 'SVG', 'EPS'])
            self.supported_formats['output'].extend(['PSD', 'SVG', 'EPS'])
        
        # Color modes
        self.color_modes = ['RGB', 'RGBA', 'L', 'LA', 'CMYK', '1', 'P']
        
        # Format-specific options
        self.format_options = {
            'JPEG': {
                'quality': (1, 100, 90),
                'progressive': (False, True, False),
                'optimize': (False, True, True),
                'subsampling': ['4:4:4', '4:2:2', '4:2:0', '4:1:1']
            },
            'PNG': {
                'compression': (0, 9, 6),
                'optimize': (False, True, True),
                'bits': [8, 16, 24, 32]
            },
            'WEBP': {
                'quality': (1, 100, 80),
                'lossless': (False, True, False),
                'method': (0, 6, 4)
            },
            'TIFF': {
                'compression': ['none', 'tiff_lzw', 'tiff_deflate', 'tiff_adobe_deflate', 'jpeg'],
                'resolution': (72, 1200, 300)
            },
            'GIF': {
                'optimize': (False, True, True),
                'transparency': (False, True, True)
            },
            'PDF': {
                'resolution': (72, 1200, 300),
                'quality': (1, 100, 90)
            }
        }
    
    def convert_image(self, image, output_format, **kwargs):
        """
        Convert an image to a different format.
        
        Args:
            image: PIL Image object
            output_format: Target format (e.g., 'PNG', 'JPEG')
            **kwargs: Format-specific options
            
        Returns:
            Converted PIL Image
        """
        # Make a copy of the image to avoid modifying the original
        converted = image.copy()
        
        # Normalize format name
        output_format = output_format.upper()
        if output_format == 'JPG':
            output_format = 'JPEG'
        
        # Check if format is supported
        if output_format not in self.supported_formats['output']:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Convert color mode if specified
        color_mode = kwargs.get('color_mode')
        if color_mode and color_mode != converted.mode:
            # Handle special cases
            if color_mode == 'RGBA' and converted.mode == 'RGB':
                # Add alpha channel
                r, g, b = converted.split()
                a = Image.new('L', converted.size, 255)  # Fully opaque
                converted = Image.merge('RGBA', (r, g, b, a))
            elif color_mode == 'RGB' and converted.mode == 'RGBA':
                # Remove alpha channel
                if kwargs.get('background_color'):
                    bg_color = kwargs.get('background_color')
                    background = Image.new('RGB', converted.size, bg_color)
                    background.paste(converted, mask=converted.split()[3])
                    converted = background
                else:
                    converted = converted.convert('RGB')
            else:
                # Standard conversion
                try:
                    converted = converted.convert(color_mode)
                except ValueError:
                    # Some conversions are not directly supported by PIL
                    # Try intermediate conversion
                    if converted.mode == 'CMYK' and color_mode == 'RGBA':
                        converted = converted.convert('RGB').convert('RGBA')
                    elif converted.mode == 'RGBA' and color_mode == 'CMYK':
                        converted = converted.convert('RGB').convert('CMYK')
                    else:
                        raise ValueError(f"Cannot convert from {converted.mode} to {color_mode}")
        
        # Apply format-specific preprocessing
        if output_format == 'JPEG' and converted.mode == 'RGBA':
            # JPEG doesn't support alpha, so we need to flatten
            bg_color = kwargs.get('background_color', (255, 255, 255))
            background = Image.new('RGB', converted.size, bg_color)
            background.paste(converted, mask=converted.split()[3])
            converted = background
        elif output_format == 'PNG' and converted.mode == 'CMYK':
            # PNG doesn't support CMYK, convert to RGBA
            converted = converted.convert('RGBA')
        elif output_format == 'GIF' and converted.mode not in ['RGB', 'RGBA', 'P']:
            # GIF requires RGB, RGBA or P mode
            converted = converted.convert('RGBA')
        
        # Set format attribute
        converted.format = output_format
        
        return converted
    
    def save_converted_image(self, image, file_path, **kwargs):
        """
        Save a converted image to a file with format-specific options.
        
        Args:
            image: PIL Image object
            file_path: Path to save the image
            **kwargs: Format-specific options
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine format from file extension if not specified
            output_format = kwargs.get('format')
            if not output_format:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.jpg':
                    output_format = 'JPEG'
                elif ext == '.tif':
                    output_format = 'TIFF'
                else:
                    output_format = ext[1:].upper()
            
            # Normalize format name
            output_format = output_format.upper()
            if output_format == 'JPG':
                output_format = 'JPEG'
            
            # Convert image if needed
            if hasattr(image, 'format') and image.format != output_format:
                image = self.convert_image(image, output_format, **kwargs)
            
            # Prepare save arguments
            save_args = {}
            
            # Format-specific options
            if output_format == 'JPEG':
                save_args['quality'] = kwargs.get('quality', self.settings['jpeg_quality'])
                save_args['optimize'] = kwargs.get('optimize', True)
                save_args['progressive'] = kwargs.get('progressive', False)
                
                # Handle subsampling
                subsampling = kwargs.get('subsampling')
                if subsampling:
                    if subsampling == '4:4:4':
                        save_args['subsampling'] = 0
                    elif subsampling == '4:2:2':
                        save_args['subsampling'] = 1
                    elif subsampling == '4:2:0':
                        save_args['subsampling'] = 2
                    elif subsampling == '4:1:1':
                        save_args['subsampling'] = 3
            
            elif output_format == 'PNG':
                save_args['optimize'] = kwargs.get('optimize', True)
                
                # Compression level
                compression = kwargs.get('compression', self.settings['png_compression'])
                if compression is not None:
                    save_args['compress_level'] = compression
                
                # Bits (via quantization for lower bit depths)
                bits = kwargs.get('bits')
                if bits and bits < 24 and image.mode in ['RGB', 'RGBA']:
                    if bits == 8:
                        # Convert to 8-bit palette
                        if OPENCV_AVAILABLE:
                            # Use OpenCV for better quantization
                            img_array = np.array(image)
                            if image.mode == 'RGBA':
                                # Preserve alpha channel
                                rgb = img_array[:, :, :3]
                                alpha = img_array[:, :, 3]
                                
                                # Quantize RGB channels
                                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                quantized = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                                
                                # Recombine with alpha
                                img_array[:, :, :3] = quantized
                            else:
                                # Just quantize RGB
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                            
                            image = Image.fromarray(img_array)
                        else:
                            # Use PIL's quantize
                            if image.mode == 'RGBA':
                                # We need to handle alpha separately
                                r, g, b, a = image.split()
                                rgb = Image.merge('RGB', (r, g, b))
                                rgb = rgb.quantize(256)
                                
                                # Convert back to RGBA
                                rgb = rgb.convert('RGB')
                                r, g, b = rgb.split()
                                image = Image.merge('RGBA', (r, g, b, a))
                            else:
                                image = image.quantize(256)
            
            elif output_format == 'WEBP':
                save_args['quality'] = kwargs.get('quality', self.settings['webp_quality'])
                save_args['lossless'] = kwargs.get('lossless', self.settings['webp_lossless'])
                
                # Method (0=fastest, 6=best quality)
                method = kwargs.get('method')
                if method is not None:
                    save_args['method'] = method
            
            elif output_format == 'TIFF':
                compression = kwargs.get('compression', self.settings['tiff_compression'])
                if compression:
                    save_args['compression'] = compression
                
                # Resolution (DPI)
                resolution = kwargs.get('resolution', self.settings['default_dpi'])
                if resolution:
                    save_args['dpi'] = (resolution, resolution)
            
            elif output_format == 'GIF':
                save_args['optimize'] = kwargs.get('optimize', self.settings['gif_optimize'])
                
                # Transparency
                if kwargs.get('transparency', True) and image.mode == 'RGBA':
                    # Find the most common fully transparent pixel
                    alpha = image.split()[3]
                    transparent_pixels = [i for i in range(256) if alpha.point(lambda x: x == i).getbbox() is None]
                    if transparent_pixels:
                        save_args['transparency'] = transparent_pixels[0]
            
            elif output_format == 'PDF':
                # PDF requires special handling
                return self._save_as_pdf(image, file_path, **kwargs)
            
            # Save the image
            image.save(file_path, format=output_format, **save_args)
            
            return True
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error saving converted image: {str(e)}")
            else:
                print(f"Error saving converted image: {str(e)}")
            return False
    
    def _save_as_pdf(self, image, file_path, **kwargs):
        """
        Save an image as PDF.
        
        Args:
            image: PIL Image object
            file_path: Path to save the PDF
            **kwargs: PDF-specific options
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if reportlab is available for better PDF creation
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.lib.utils import ImageReader
                REPORTLAB_AVAILABLE = True
            except ImportError:
                REPORTLAB_AVAILABLE = False
            
            if REPORTLAB_AVAILABLE:
                # Use reportlab for better PDF creation
                # Get page size
                page_size = kwargs.get('page_size', 'A4')
                if page_size == 'A4':
                    width, height = A4
                elif page_size == 'letter':
                    width, height = letter
                else:
                    # Custom size
                    width = kwargs.get('width', 595)  # A4 width in points
                    height = kwargs.get('height', 842)  # A4 height in points
                
                # Create PDF canvas
                c = canvas.Canvas(file_path, pagesize=(width, height))
                
                # Calculate image placement to center it
                img_width, img_height = image.size
                
                # Convert to RGB if needed
                if image.mode == 'RGBA':
                    # Create a white background
                    bg_color = kwargs.get('background_color', (255, 255, 255))
                    background = Image.new('RGB', image.size, bg_color)
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Calculate scaling to fit the page
                scale_x = width / img_width
                scale_y = height / img_height
                scale = min(scale_x, scale_y) * 0.9  # 90% to leave some margin
                
                # Calculate position to center the image
                x = (width - img_width * scale) / 2
                y = (height - img_height * scale) / 2
                
                # Draw the image
                c.drawImage(ImageReader(image), x, y, width=img_width * scale, height=img_height * scale)
                
                # Add metadata
                title = kwargs.get('title', 'Image')
                author = kwargs.get('author', 'Image Processor')
                subject = kwargs.get('subject', 'Converted Image')
                
                c.setTitle(title)
                c.setAuthor(author)
                c.setSubject(subject)
                
                # Save the PDF
                c.save()
                
            else:
                # Use PIL's PDF support (more limited)
                # Convert to RGB if needed
                if image.mode == 'RGBA':
                    # Create a white background
                    bg_color = kwargs.get('background_color', (255, 255, 255))
                    background = Image.new('RGB', image.size, bg_color)
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resolution (DPI)
                resolution = kwargs.get('resolution', self.settings['default_dpi'])
                dpi = (resolution, resolution)
                
                # Save as PDF
                image.save(file_path, 'PDF', resolution=dpi)
            
            return True
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error saving as PDF: {str(e)}")
            else:
                print(f"Error saving as PDF: {str(e)}")
            return False
    
    def convert_color_mode(self, image, target_mode, **kwargs):
        """
        Convert an image to a different color mode.
        
        Args:
            image: PIL Image object
            target_mode: Target color mode (e.g., 'RGB', 'RGBA', 'L')
            **kwargs: Additional options
            
        Returns:
            Converted PIL Image
        """
        # Make a copy of the image to avoid modifying the original
        converted = image.copy()
        
        # Check if conversion is needed
        if converted.mode == target_mode:
            return converted
        
        try:
            # Handle special cases
            if converted.mode == 'RGB' and target_mode == 'RGBA':
                # Add alpha channel
                r, g, b = converted.split()
                a = Image.new('L', converted.size, 255)  # Fully opaque
                converted = Image.merge('RGBA', (r, g, b, a))
                
            elif converted.mode == 'RGBA' and target_mode == 'RGB':
                # Remove alpha channel with background color
                bg_color = kwargs.get('background_color', (255, 255, 255))
                background = Image.new('RGB', converted.size, bg_color)
                background.paste(converted, mask=converted.split()[3])
                converted = background
                
            elif converted.mode == 'RGB' and target_mode == 'L':
                # Convert to grayscale
                if OPENCV_AVAILABLE and kwargs.get('use_opencv', True):
                    # Use OpenCV for better grayscale conversion
                    img_array = np.array(converted)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    converted = Image.fromarray(gray, 'L')
                else:
                    # Use PIL's conversion
                    converted = converted.convert('L')
                
            elif converted.mode == 'RGBA' and target_mode == 'L':
                # Convert to grayscale with alpha
                r, g, b, a = converted.split()
                if OPENCV_AVAILABLE and kwargs.get('use_opencv', True):
                    # Use OpenCV for better grayscale conversion
                    rgb = Image.merge('RGB', (r, g, b))
                    img_array = np.array(rgb)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    l = Image.fromarray(gray, 'L')
                else:
                    # Use PIL's conversion
                    rgb = Image.merge('RGB', (r, g, b))
                    l = rgb.convert('L')
                
                # Merge with original alpha
                converted = Image.merge('LA', (l, a))
                
            elif converted.mode == 'L' and target_mode == 'RGB':
                # Convert grayscale to RGB
                converted = converted.convert('RGB')
                
            elif converted.mode == 'L' and target_mode == 'RGBA':
                # Convert grayscale to RGBA
                l = converted
                a = Image.new('L', converted.size, 255)  # Fully opaque
                converted = l.convert('RGB')
                r, g, b = converted.split()
                converted = Image.merge('RGBA', (r, g, b, a))
                
            elif converted.mode == 'LA' and target_mode == 'RGBA':
                # Convert grayscale with alpha to RGBA
                l, a = converted.split()
                r = g = b = l.copy()
                converted = Image.merge('RGBA', (r, g, b, a))
                
            elif converted.mode == 'P' and target_mode in ['RGB', 'RGBA']:
                # Convert palette to RGB/RGBA
                converted = converted.convert(target_mode)
                
            elif target_mode == 'P':
                # Convert to palette (indexed color)
                if converted.mode != 'RGB' and converted.mode != 'RGBA':
                    converted = converted.convert('RGB')
                
                # Get palette options
                colors = kwargs.get('colors', 256)
                dither = kwargs.get('dither', True)
                
                # Use quantize for conversion to palette
                if dither:
                    converted = converted.quantize(colors=colors)
                else:
                    converted = converted.quantize(colors=colors, dither=0)
                
            elif target_mode == 'CMYK':
                # Convert to CMYK
                if converted.mode != 'RGB':
                    converted = converted.convert('RGB')
                converted = converted.convert('CMYK')
                
            elif target_mode == '1':
                # Convert to 1-bit (black and white)
                if converted.mode != 'L':
                    converted = converted.convert('L')
                
                # Get threshold
                threshold = kwargs.get('threshold', 128)
                converted = converted.point(lambda x: 255 if x > threshold else 0, '1')
                
            else:
                # Standard conversion
                converted = converted.convert(target_mode)
            
            return converted
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error converting color mode: {str(e)}")
            else:
                print(f"Error converting color mode: {str(e)}")
            return image  # Return original on error
    
    def optimize_image(self, image, target_format, **kwargs):
        """
        Optimize an image for a specific format.
        
        Args:
            image: PIL Image object
            target_format: Target format (e.g., 'PNG', 'JPEG')
            **kwargs: Format-specific optimization options
            
        Returns:
            Optimized PIL Image and recommended save options
        """
        # Make a copy of the image to avoid modifying the original
        optimized = image.copy()
        
        # Normalize format name
        target_format = target_format.upper()
        if target_format == 'JPG':
            target_format = 'JPEG'
        
        # Prepare save options
        save_options = {}
        
        try:
            # Format-specific optimizations
            if target_format == 'JPEG':
                # Convert to RGB if needed
                if optimized.mode != 'RGB':
                    bg_color = kwargs.get('background_color', (255, 255, 255))
                    if optimized.mode == 'RGBA':
                        background = Image.new('RGB', optimized.size, bg_color)
                        background.paste(optimized, mask=optimized.split()[3])
                        optimized = background
                    else:
                        optimized = optimized.convert('RGB')
                
                # Get quality setting
                quality = kwargs.get('quality', self.settings['jpeg_quality'])
                save_options['quality'] = quality
                save_options['optimize'] = True
                
                # Progressive JPEG
                if kwargs.get('progressive', False):
                    save_options['progressive'] = True
                
                # Subsampling
                subsampling = kwargs.get('subsampling')
                if subsampling:
                    if subsampling == '4:4:4':
                        save_options['subsampling'] = 0
                    elif subsampling == '4:2:2':
                        save_options['subsampling'] = 1
                    elif subsampling == '4:2:0':
                        save_options['subsampling'] = 2
                    elif subsampling == '4:1:1':
                        save_options['subsampling'] = 3
                
            elif target_format == 'PNG':
                # Optimize for PNG
                
                # Check if image has transparency
                has_transparency = optimized.mode == 'RGBA' and any(
                    pixel[3] < 255 for pixel in optimized.getdata()
                )
                
                if not has_transparency and optimized.mode == 'RGBA':
                    # Convert to RGB if no transparency
                    optimized = optimized.convert('RGB')
                
                # Get compression level
                compression = kwargs.get('compression', self.settings['png_compression'])
                save_options['compress_level'] = compression
                save_options['optimize'] = True
                
                # Reduce colors if requested
                bits = kwargs.get('bits')
                if bits and bits < 24:
                    if bits == 8:
                        # Convert to 8-bit palette
                        if optimized.mode in ['RGB', 'RGBA']:
                            # Preserve alpha if needed
                            has_alpha = optimized.mode == 'RGBA'
                            alpha = None
                            
                            if has_alpha:
                                # Extract alpha channel
                                alpha = optimized.split()[3]
                                
                            # Quantize to 256 colors
                            if OPENCV_AVAILABLE:
                                # Use OpenCV for better quantization
                                img_array = np.array(optimized)
                                if has_alpha:
                                    # Process RGB channels only
                                    rgb = img_array[:, :, :3]
                                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                    
                                    # Apply color quantization
                                    z = rgb.reshape((-1, 3))
                                    z = np.float32(z)
                                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                                    K = 256
                                    _, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                                    
                                    # Convert back to uint8
                                    center = np.uint8(center)
                                    res = center[label.flatten()]
                                    quantized = res.reshape((img_array.shape[0], img_array.shape[1], 3))
                                    
                                    # Convert back to RGB
                                    quantized = cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB)
                                    
                                    # Create new image
                                    optimized = Image.fromarray(quantized, 'RGB')
                                    
                                    # Reapply alpha if needed
                                    if alpha is not None:
                                        optimized = optimized.convert('RGBA')
                                        r, g, b, _ = optimized.split()
                                        optimized = Image.merge('RGBA', (r, g, b, alpha))
                                else:
                                    # Just quantize RGB
                                    optimized = optimized.quantize(256)
                            else:
                                # Use PIL's quantize
                                if has_alpha:
                                    # We need to handle alpha separately
                                    r, g, b, a = optimized.split()
                                    rgb = Image.merge('RGB', (r, g, b))
                                    rgb = rgb.quantize(256)
                                    
                                    # Convert back to RGBA
                                    rgb = rgb.convert('RGB')
                                    r, g, b = rgb.split()
                                    optimized = Image.merge('RGBA', (r, g, b, a))
                                else:
                                    optimized = optimized.quantize(256)
                
            elif target_format == 'WEBP':
                # Optimize for WebP
                
                # Quality setting
                quality = kwargs.get('quality', self.settings['webp_quality'])
                save_options['quality'] = quality
                
                # Lossless option
                lossless = kwargs.get('lossless', self.settings['webp_lossless'])
                save_options['lossless'] = lossless
                
                # Method (0=fastest, 6=best quality)
                method = kwargs.get('method', 4)
                save_options['method'] = method
                
            elif target_format == 'GIF':
                # Optimize for GIF
                
                # Convert to palette mode if not already
                if optimized.mode != 'P':
                    # Extract alpha channel if present
                    has_alpha = optimized.mode == 'RGBA'
                    alpha = None
                    
                    if has_alpha:
                        # Extract alpha channel
                        alpha = optimized.split()[3]
                    
                    # Convert to palette
                    colors = kwargs.get('colors', 256)
                    dither = kwargs.get('dither', True)
                    
                    if dither:
                        optimized = optimized.quantize(colors=colors)
                    else:
                        optimized = optimized.quantize(colors=colors, dither=0)
                    
                    # Handle transparency
                    if has_alpha and kwargs.get('transparency', True):
                        # Find transparent pixels
                        transparent_pixels = []
                        for y in range(alpha.height):
                            for x in range(alpha.width):
                                if alpha.getpixel((x, y)) < 128:
                                    transparent_pixels.append((x, y))
                        
                        if transparent_pixels:
                            # Get the palette
                            palette = optimized.getpalette()
                            
                            # Find an unused color index for transparency
                            used_indices = set(optimized.getdata())
                            transparent_index = None
                            
                            for i in range(256):
                                if i not in used_indices:
                                    transparent_index = i
                                    break
                            
                            if transparent_index is not None:
                                # Set transparent pixels
                                pixdata = optimized.load()
                                for x, y in transparent_pixels:
                                    pixdata[x, y] = transparent_index
                                
                                # Set transparency in the palette
                                optimized.info['transparency'] = transparent_index
                                save_options['transparency'] = transparent_index
                
                # Optimize option
                save_options['optimize'] = kwargs.get('optimize', self.settings['gif_optimize'])
                
            elif target_format == 'TIFF':
                # Optimize for TIFF
                
                # Compression option
                compression = kwargs.get('compression', self.settings['tiff_compression'])
                save_options['compression'] = compression
                
                # Resolution (DPI)
                resolution = kwargs.get('resolution', self.settings['default_dpi'])
                save_options['dpi'] = (resolution, resolution)
                
            elif target_format == 'BMP':
                # BMP has limited options
                pass
                
            # Return optimized image and save options
            return optimized, save_options
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error optimizing image: {str(e)}")
            else:
                print(f"Error optimizing image: {str(e)}")
            return image, {}  # Return original on error
    
    def resize_image(self, image, width, height, **kwargs):
        """
        Resize an image with various options.
        
        Args:
            image: PIL Image object
            width: Target width
            height: Target height
            **kwargs: Additional options
            
        Returns:
            Resized PIL Image
        """
        # Make a copy of the image to avoid modifying the original
        resized = image.copy()
        
        try:
            # Get resize options
            maintain_aspect = kwargs.get('maintain_aspect', True)
            resize_mode = kwargs.get('resize_mode', 'fit')  # 'fit', 'fill', 'stretch'
            resampling = kwargs.get('resampling', 'lanczos')  # 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos'
            
            # Convert resampling string to PIL constant
            resampling_map = {
                'nearest': Image.NEAREST,
                'box': Image.BOX,
                'bilinear': Image.BILINEAR,
                'hamming': Image.HAMMING,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS
            }
            resample = resampling_map.get(resampling.lower(), Image.LANCZOS)
            
            # Calculate new dimensions
            orig_width, orig_height = resized.size
            
            if maintain_aspect:
                # Calculate aspect ratios
                orig_aspect = orig_width / orig_height
                target_aspect = width / height
                
                if resize_mode == 'fit':
                    # Fit entire image within target dimensions
                    if orig_aspect > target_aspect:
                        # Image is wider than target, constrain width
                        new_width = width
                        new_height = int(width / orig_aspect)
                    else:
                        # Image is taller than target, constrain height
                        new_width = int(height * orig_aspect)
                        new_height = height
                        
                elif resize_mode == 'fill':
                    # Fill target dimensions, cropping if necessary
                    if orig_aspect > target_aspect:
                        # Image is wider than target, constrain height
                        new_width = int(height * orig_aspect)
                        new_height = height
                    else:
                        # Image is taller than target, constrain width
                        new_width = width
                        new_height = int(width / orig_aspect)
                    
                    # Resize first
                    resized = resized.resize((new_width, new_height), resample)
                    
                    # Then crop to target dimensions
                    left = (new_width - width) // 2
                    top = (new_height - height) // 2
                    right = left + width
                    bottom = top + height
                    
                    resized = resized.crop((left, top, right, bottom))
                    return resized
            else:
                # Stretch to fit target dimensions
                new_width = width
                new_height = height
            
            # Perform the resize
            if OPENCV_AVAILABLE and kwargs.get('use_opencv', False):
                # Use OpenCV for resizing
                img_array = np.array(resized)
                
                # Determine interpolation method
                if resampling == 'nearest':
                    interpolation = cv2.INTER_NEAREST
                elif resampling == 'bilinear':
                    interpolation = cv2.INTER_LINEAR
                elif resampling == 'bicubic':
                    interpolation = cv2.INTER_CUBIC
                else:
                    interpolation = cv2.INTER_LANCZOS4
                
                # Resize the image
                if resized.mode == 'RGBA':
                    # Handle alpha channel separately
                    rgb = img_array[:, :, :3]
                    alpha = img_array[:, :, 3]
                    
                    rgb_resized = cv2.resize(rgb, (new_width, new_height), interpolation=interpolation)
                    alpha_resized = cv2.resize(alpha, (new_width, new_height), interpolation=interpolation)
                    
                    # Combine channels
                    resized_array = np.zeros((new_height, new_width, 4), dtype=np.uint8)
                    resized_array[:, :, :3] = rgb_resized
                    resized_array[:, :, 3] = alpha_resized
                    
                    resized = Image.fromarray(resized_array)
                else:
                    # Resize directly
                    resized_array = cv2.resize(img_array, (new_width, new_height), interpolation=interpolation)
                    resized = Image.fromarray(resized_array)
            else:
                # Use PIL for resizing
                resized = resized.resize((new_width, new_height), resample)
            
            return resized
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error resizing image: {str(e)}")
            else:
                print(f"Error resizing image: {str(e)}")
            return image  # Return original on error
    
    def create_converter_dialog(self, parent, image):
        """
        Create a dialog for converting an image.
        
        Args:
            parent: Parent window
            image: PIL Image to convert
            
        Returns:
            Converter dialog
        """
        # Create dialog
        dialog = tk.Toplevel(parent)
        dialog.title("Convert Image")
        dialog.geometry("600x700")
        dialog.minsize(600, 700)
        
        # Make it modal
        dialog.transient(parent)
        dialog.grab_set()
        
        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image info
        info_frame = ttk.LabelFrame(main_frame, text="Image Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current format and dimensions
        img_width, img_height = image.size
        ttk.Label(info_frame, text=f"Current Format: {image.format or 'Unknown'}").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text=f"Current Mode: {image.mode}").pack(anchor=tk.W, pady=2)
        ttk.Label(info_frame, text=f"Dimensions: {img_width} × {img_height} pixels").pack(anchor=tk.W, pady=2)
        
        # Create a notebook for different conversion options
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Format conversion tab
        format_tab = ttk.Frame(notebook, padding=10)
        notebook.add(format_tab, text="Format")
        
        # Format selection
        format_frame = ttk.Frame(format_tab)
        format_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(format_frame, text="Target Format:").pack(side=tk.LEFT)
        
        format_var = tk.StringVar(value=self.settings['default_format'])
        format_combo = ttk.Combobox(format_frame, textvariable=format_var, width=10)
        format_combo['values'] = self.supported_formats['output']
        format_combo.pack(side=tk.LEFT, padx=5)
        
        # Format-specific options frame
        format_options_frame = ttk.LabelFrame(format_tab, text="Format Options")
        format_options_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # JPEG options
        jpeg_frame = ttk.Frame(format_options_frame)
        
        ttk.Label(jpeg_frame, text="Quality:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        jpeg_quality_var = tk.IntVar(value=self.settings['jpeg_quality'])
        jpeg_quality_scale = ttk.Scale(jpeg_frame, from_=1, to=100, variable=jpeg_quality_var, 
                                      orient=tk.HORIZONTAL)
        jpeg_quality_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        jpeg_quality_label = ttk.Label(jpeg_frame, text=f"{jpeg_quality_var.get()}%")
        jpeg_quality_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Update quality label
        def update_jpeg_quality_label(*args):
            jpeg_quality_label.config(text=f"{jpeg_quality_var.get()}%")
        
        jpeg_quality_var.trace_add("write", update_jpeg_quality_label)
        
        # Progressive JPEG
        jpeg_progressive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(jpeg_frame, text="Progressive", 
                       variable=jpeg_progressive_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Optimize
        jpeg_optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(jpeg_frame, text="Optimize", 
                       variable=jpeg_optimize_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Subsampling
        ttk.Label(jpeg_frame, text="Subsampling:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        jpeg_subsampling_var = tk.StringVar(value="4:2:0")
        jpeg_subsampling_combo = ttk.Combobox(jpeg_frame, textvariable=jpeg_subsampling_var, width=10)
        jpeg_subsampling_combo['values'] = ['4:4:4', '4:2:2', '4:2:0', '4:1:1']
        jpeg_subsampling_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # PNG options
        png_frame = ttk.Frame(format_options_frame)
        
        ttk.Label(png_frame, text="Compression:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        png_compression_var = tk.IntVar(value=self.settings['png_compression'])
        png_compression_scale = ttk.Scale(png_frame, from_=0, to=9, variable=png_compression_var, 
                                         orient=tk.HORIZONTAL)
        png_compression_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        png_compression_label = ttk.Label(png_frame, text=str(png_compression_var.get()))
        png_compression_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Update compression label
        def update_png_compression_label(*args):
            png_compression_label.config(text=str(png_compression_var.get()))
        
        png_compression_var.trace_add("write", update_png_compression_label)
        
        # Optimize
        png_optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(png_frame, text="Optimize", 
                       variable=png_optimize_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Bit depth
        ttk.Label(png_frame, text="Bit Depth:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        png_bits_var = tk.IntVar(value=32)
        png_bits_combo = ttk.Combobox(png_frame, textvariable=png_bits_var, width=10)
        png_bits_combo['values'] = [8, 24, 32]
        png_bits_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # WebP options
        webp_frame = ttk.Frame(format_options_frame)
        
        ttk.Label(webp_frame, text="Quality:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        webp_quality_var = tk.IntVar(value=self.settings['webp_quality'])
        webp_quality_scale = ttk.Scale(webp_frame, from_=1, to=100, variable=webp_quality_var, 
                                      orient=tk.HORIZONTAL)
        webp_quality_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        webp_quality_label = ttk.Label(webp_frame, text=f"{webp_quality_var.get()}%")
        webp_quality_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Update quality label
        def update_webp_quality_label(*args):
            webp_quality_label.config(text=f"{webp_quality_var.get()}%")
        
        webp_quality_var.trace_add("write", update_webp_quality_label)
        
        # Lossless
        webp_lossless_var = tk.BooleanVar(value=self.settings['webp_lossless'])
        ttk.Checkbutton(webp_frame, text="Lossless", 
                       variable=webp_lossless_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Method
        ttk.Label(webp_frame, text="Method:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        webp_method_var = tk.IntVar(value=4)
        webp_method_scale = ttk.Scale(webp_frame, from_=0, to=6, variable=webp_method_var, 
                                     orient=tk.HORIZONTAL)
        webp_method_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        webp_method_label = ttk.Label(webp_frame, text=str(webp_method_var.get()))
        webp_method_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Update method label
        def update_webp_method_label(*args):
            webp_method_label.config(text=str(webp_method_var.get()))
        
        webp_method_var.trace_add("write", update_webp_method_label)
        
        # TIFF options
        tiff_frame = ttk.Frame(format_options_frame)
        
        ttk.Label(tiff_frame, text="Compression:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        tiff_compression_var = tk.StringVar(value=self.settings['tiff_compression'])
        tiff_compression_combo = ttk.Combobox(tiff_frame, textvariable=tiff_compression_var, width=15)
        tiff_compression_combo['values'] = ['none', 'tiff_lzw', 'tiff_deflate', 'tiff_adobe_deflate', 'jpeg']
        tiff_compression_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Resolution (DPI)
        ttk.Label(tiff_frame, text="Resolution (DPI):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        tiff_dpi_var = tk.IntVar(value=self.settings['default_dpi'])
        tiff_dpi_combo = ttk.Combobox(tiff_frame, textvariable=tiff_dpi_var, width=10)
        tiff_dpi_combo['values'] = [72, 96, 150, 300, 600]
        tiff_dpi_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # GIF options
        gif_frame = ttk.Frame(format_options_frame)
        
        # Optimize
        gif_optimize_var = tk.BooleanVar(value=self.settings['gif_optimize'])
        ttk.Checkbutton(gif_frame, text="Optimize", 
                       variable=gif_optimize_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Transparency
        gif_transparency_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(gif_frame, text="Preserve Transparency", 
                       variable=gif_transparency_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # PDF options
        pdf_frame = ttk.Frame(format_options_frame)
        
        # Page size
        ttk.Label(pdf_frame, text="Page Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        pdf_page_var = tk.StringVar(value="A4")
        pdf_page_combo = ttk.Combobox(pdf_frame, textvariable=pdf_page_var, width=10)
        pdf_page_combo['values'] = ['A4', 'letter', 'custom']
        pdf_page_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Resolution (DPI)
        ttk.Label(pdf_frame, text="Resolution (DPI):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        pdf_dpi_var = tk.IntVar(value=self.settings['default_dpi'])
        pdf_dpi_combo = ttk.Combobox(pdf_frame, textvariable=pdf_dpi_var, width=10)
        pdf_dpi_combo['values'] = [72, 96, 150, 300, 600]
        pdf_dpi_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Function to show the appropriate options frame based on format
        def show_format_options(*args):
            # Hide all frames
            for frame in [jpeg_frame, png_frame, webp_frame, tiff_frame, gif_frame, pdf_frame]:
                for widget in frame.winfo_children():
                    widget.grid_forget()
                frame.pack_forget()
            
            # Show the appropriate frame
            format_str = format_var.get()
            if format_str == 'JPEG' or format_str == 'JPG':
                jpeg_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                for widget in jpeg_frame.winfo_children():
                    widget.grid(sticky=tk.W)
            elif format_str == 'PNG':
                png_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                for widget in png_frame.winfo_children():
                    widget.grid(sticky=tk.W)
            elif format_str == 'WEBP':
                webp_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                for widget in webp_frame.winfo_children():
                    widget.grid(sticky=tk.W)
            elif format_str == 'TIFF':
                tiff_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                for widget in tiff_frame.winfo_children():
                    widget.grid(sticky=tk.W)
            elif format_str == 'GIF':
                gif_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                for widget in gif_frame.winfo_children():
                    widget.grid(sticky=tk.W)
            elif format_str == 'PDF':
                pdf_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                for widget in pdf_frame.winfo_children():
                    widget.grid(sticky=tk.W)
        
        # Bind format change
        format_var.trace_add("write", show_format_options)
        
        # Color mode tab
        color_tab = ttk.Frame(notebook, padding=10)
        notebook.add(color_tab, text="Color Mode")
        
        # Color mode selection
        color_frame = ttk.Frame(color_tab)
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(color_frame, text="Target Color Mode:").pack(side=tk.LEFT)
        
        color_mode_var = tk.StringVar(value=image.mode)
        color_mode_combo = ttk.Combobox(color_frame, textvariable=color_mode_var, width=10)
        color_mode_combo['values'] = self.color_modes
        color_mode_combo.pack(side=tk.LEFT, padx=5)
        
        # Color mode options frame
        color_options_frame = ttk.LabelFrame(color_tab, text="Color Mode Options")
        color_options_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Background color for RGB conversion
        bg_color_frame = ttk.Frame(color_options_frame)
        bg_color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(bg_color_frame, text="Background Color:").pack(side=tk.LEFT)
        
        bg_color_var = tk.StringVar(value="#ffffff")
        bg_color_preview = tk.Canvas(bg_color_frame, width=30, height=20, bg=bg_color_var.get())
        bg_color_preview.pack(side=tk.LEFT, padx=5)
        
        # Color picker button
        def pick_bg_color():
            color = colorchooser.askcolor(title="Choose Background Color", 
                                         initialcolor=bg_color_var.get())
            if color[1]:
                bg_color_var.set(color[1])
                bg_color_preview.config(bg=color[1])
        
        ttk.Button(bg_color_frame, text="Pick Color", command=pick_bg_color).pack(side=tk.LEFT)
        
        # Grayscale conversion options
        grayscale_frame = ttk.Frame(color_options_frame)
        grayscale_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(grayscale_frame, text="Grayscale Method:").pack(side=tk.LEFT)
        
        grayscale_method_var = tk.StringVar(value="luminosity")
        grayscale_method_combo = ttk.Combobox(grayscale_frame, textvariable=grayscale_method_var, width=15)
        grayscale_method_combo['values'] = ['average', 'luminosity', 'lightness']
        grayscale_method_combo.pack(side=tk.LEFT, padx=5)
        
        # Black and white threshold
        bw_frame = ttk.Frame(color_options_frame)
        bw_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(bw_frame, text="B&W Threshold:").pack(side=tk.LEFT)
        
        bw_threshold_var = tk.IntVar(value=128)
        bw_threshold_scale = ttk.Scale(bw_frame, from_=0, to=255, variable=bw_threshold_var, 
                                      orient=tk.HORIZONTAL)
        bw_threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        bw_threshold_label = ttk.Label(bw_frame, text=str(bw_threshold_var.get()))
        bw_threshold_label.pack(side=tk.LEFT)
        
        # Update threshold label
        def update_bw_threshold_label(*args):
            bw_threshold_label.config(text=str(bw_threshold_var.get()))
        
        bw_threshold_var.trace_add("write", update_bw_threshold_label)
        
        # Palette options
        palette_frame = ttk.Frame(color_options_frame)
        palette_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(palette_frame, text="Palette Colors:").pack(side=tk.LEFT)
        
        palette_colors_var = tk.IntVar(value=256)
        palette_colors_combo = ttk.Combobox(palette_frame, textvariable=palette_colors_var, width=10)
        palette_colors_combo['values'] = [2, 4, 8, 16, 32, 64, 128, 256]
        palette_colors_combo.pack(side=tk.LEFT, padx=5)
        
        # Dither option
        palette_dither_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(palette_frame, text="Dither", 
                       variable=palette_dither_var).pack(side=tk.LEFT, padx=10)
        
        # Function to show appropriate color mode options
        def show_color_options(*args):
            # Hide all frames
            for frame in [bg_color_frame, grayscale_frame, bw_frame, palette_frame]:
                frame.pack_forget()
            
            # Show the appropriate frames
            mode = color_mode_var.get()
            
            if mode == 'RGB' and image.mode == 'RGBA':
                # Show background color option when converting from RGBA to RGB
                bg_color_frame.pack(fill=tk.X, pady=5)
            
            if mode == 'L':
                # Grayscale options
                grayscale_frame.pack(fill=tk.X, pady=5)
            
            if mode == '1':
                # Black and white options
                bw_frame.pack(fill=tk.X, pady=5)
            
            if mode == 'P':
                # Palette options
                palette_frame.pack(fill=tk.X, pady=5)
        
        # Bind color mode change
        color_mode_var.trace_add("write", show_color_options)
        
        # Resize tab
        resize_tab = ttk.Frame(notebook, padding=10)
        notebook.add(resize_tab, text="Resize")
        
        # Resize options
        resize_frame = ttk.Frame(resize_tab)
        resize_frame.pack(fill=tk.X, pady=5)
        
        resize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(resize_frame, text="Resize Image", 
                       variable=resize_var).pack(anchor=tk.W)
        
        # Dimensions
        dimensions_frame = ttk.Frame(resize_tab)
        dimensions_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dimensions_frame, text="Width:").pack(side=tk.LEFT)
        
        width_var = tk.IntVar(value=img_width)
        width_entry = ttk.Spinbox(dimensions_frame, from_=1, to=10000, width=6, textvariable=width_var)
        width_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(dimensions_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        
        height_var = tk.IntVar(value=img_height)
        height_entry = ttk.Spinbox(dimensions_frame, from_=1, to=10000, width=6, textvariable=height_var)
        height_entry.pack(side=tk.LEFT, padx=5)
        
        # Maintain aspect ratio
        aspect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(resize_tab, text="Maintain Aspect Ratio", 
                       variable=aspect_var).pack(anchor=tk.W, pady=5)
        
        # Resize mode
        mode_frame = ttk.Frame(resize_tab)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Resize Mode:").pack(side=tk.LEFT)
        
        resize_mode_var = tk.StringVar(value="fit")
        resize_mode_combo = ttk.Combobox(mode_frame, textvariable=resize_mode_var, width=10)
        resize_mode_combo['values'] = ['fit', 'fill', 'stretch']
        resize_mode_combo.pack(side=tk.LEFT, padx=5)
        
        # Resampling method
        resampling_frame = ttk.Frame(resize_tab)
        resampling_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(resampling_frame, text="Resampling:").pack(side=tk.LEFT)
        
        resampling_var = tk.StringVar(value="lanczos")
        resampling_combo = ttk.Combobox(resampling_frame, textvariable=resampling_var, width=10)
        resampling_combo['values'] = ['nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos']
        resampling_combo.pack(side=tk.LEFT, padx=5)
        
        # Common sizes
        common_frame = ttk.Frame(resize_tab)
        common_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(common_frame, text="Common Sizes:").pack(side=tk.LEFT)
        
        def set_common_size(width, height):
            width_var.set(width)
            height_var.set(height)
        
        ttk.Button(common_frame, text="HD", width=5, 
                  command=lambda: set_common_size(1280, 720)).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(common_frame, text="Full HD", width=7, 
                  command=lambda: set_common_size(1920, 1080)).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(common_frame, text="4K", width=5, 
                  command=lambda: set_common_size(3840, 2160)).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(common_frame, text="8K", width=5, 
                  command=lambda: set_common_size(7680, 4320)).pack(side=tk.LEFT, padx=2)
        
        # Update dimensions when aspect ratio is maintained
        original_aspect = img_width / img_height
        
        def update_height(*args):
            if aspect_var.get() and resize_var.get():
                new_width = width_var.get()
                new_height = int(new_width / original_aspect)
                height_var.set(new_height)
        
        def update_width(*args):
            if aspect_var.get() and resize_var.get():
                new_height = height_var.get()
                new_width = int(new_height * original_aspect)
                width_var.set(new_width)
        
        width_var.trace_add("write", update_height)
        height_var.trace_add("write", update_width)
        
        # Optimization tab
        optimize_tab = ttk.Frame(notebook, padding=10)
        notebook.add(optimize_tab, text="Optimization")
        
        # Optimization options
        optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(optimize_tab, text="Optimize Image for Target Format", 
                       variable=optimize_var).pack(anchor=tk.W, pady=5)
        
        # Metadata options
        metadata_frame = ttk.LabelFrame(optimize_tab, text="Metadata")
        metadata_frame.pack(fill=tk.X, pady=10)
        
        preserve_var = tk.BooleanVar(value=self.settings['preserve_metadata'])
        ttk.Checkbutton(metadata_frame, text="Preserve Metadata (when possible)", 
                       variable=preserve_var).pack(anchor=tk.W, pady=5)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a canvas for preview
        preview_canvas = tk.Canvas(preview_frame, bg="#f0f0f0")
        preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview image reference
        preview_image_ref = [None]
        
        # Function to update preview
        def update_preview():
            # Create a copy of the image
            preview_img = image.copy()
            
            try:
                # Apply color mode conversion if needed
                mode = color_mode_var.get()
                if mode != preview_img.mode:
                    # Get color conversion options
                    color_options = {}
                    
                    if mode == 'RGB' and preview_img.mode == 'RGBA':
                        color_options['background_color'] = bg_color_var.get()
                    
                    if mode == 'L':
                        color_options['method'] = grayscale_method_var.get()
                    
                    if mode == '1':
                        color_options['threshold'] = bw_threshold_var.get()
                    
                    if mode == 'P':
                        color_options['colors'] = palette_colors_var.get()
                        color_options['dither'] = palette_dither_var.get()
                    
                    # Convert color mode
                    preview_img = self.convert_color_mode(preview_img, mode, **color_options)
                
                # Apply resize if enabled
                if resize_var.get():
                    new_width = width_var.get()
                    new_height = height_var.get()
                    
                    # Get resize options
                    resize_options = {
                        'maintain_aspect': aspect_var.get(),
                        'resize_mode': resize_mode_var.get(),
                        'resampling': resampling_var.get()
                    }
                    
                    # Resize the image
                    preview_img = self.resize_image(preview_img, new_width, new_height, **resize_options)
                
                # Apply format-specific optimizations if enabled
                if optimize_var.get():
                    format_str = format_var.get()
                    
                    # Get format-specific options
                    format_options = {}
                    
                    if format_str == 'JPEG' or format_str == 'JPG':
                        format_options['quality'] = jpeg_quality_var.get()
                        format_options['progressive'] = jpeg_progressive_var.get()
                        format_options['optimize'] = jpeg_optimize_var.get()
                        format_options['subsampling'] = jpeg_subsampling_var.get()
                    
                    elif format_str == 'PNG':
                        format_options['compression'] = png_compression_var.get()
                        format_options['optimize'] = png_optimize_var.get()
                        format_options['bits'] = png_bits_var.get()
                    
                    elif format_str == 'WEBP':
                        format_options['quality'] = webp_quality_var.get()
                        format_options['lossless'] = webp_lossless_var.get()
                        format_options['method'] = webp_method_var.get()
                    
                    elif format_str == 'TIFF':
                        format_options['compression'] = tiff_compression_var.get()
                        format_options['resolution'] = tiff_dpi_var.get()
                    
                    elif format_str == 'GIF':
                        format_options['optimize'] = gif_optimize_var.get()
                        format_options['transparency'] = gif_transparency_var.get()
                    
                    # Optimize the image
                    preview_img, _ = self.optimize_image(preview_img, format_str, **format_options)
                
                # Calculate preview size to fit canvas
                canvas_width = preview_canvas.winfo_width()
                canvas_height = preview_canvas.winfo_height()
                
                if canvas_width <= 1 or canvas_height <= 1:
                    # Canvas not yet properly sized, use default size
                    canvas_width = 300
                    canvas_height = 200
                
                # Calculate scale factor to fit preview
                img_width, img_height = preview_img.size
                width_ratio = canvas_width / img_width
                height_ratio = canvas_height / img_height
                
                # Use the smaller ratio to ensure the entire image fits
                scale_factor = min(width_ratio, height_ratio) * 0.9  # 90% to leave some margin
                
                # Resize for preview
                preview_width = int(img_width * scale_factor)
                preview_height = int(img_height * scale_factor)
                preview_display = preview_img.resize((preview_width, preview_height), Image.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(preview_display)
                
                # Clear canvas and display image
                preview_canvas.delete("all")
                preview_canvas.create_image(
                    canvas_width // 2, canvas_height // 2,
                    image=photo, anchor=tk.CENTER
                )
                
                # Store reference to prevent garbage collection
                preview_image_ref[0] = photo
                
                # Update info label
                format_str = format_var.get()
                mode_str = color_mode_var.get()
                
                # Estimate file size
                img_byte_arr = io.BytesIO()
                
                # Save with appropriate options
                save_args = {}
                
                if format_str == 'JPEG' or format_str == 'JPG':
                    save_args['quality'] = jpeg_quality_var.get()
                    save_args['optimize'] = jpeg_optimize_var.get()
                    if jpeg_progressive_var.get():
                        save_args['progressive'] = True
                
                elif format_str == 'PNG':
                    save_args['optimize'] = png_optimize_var.get()
                    save_args['compress_level'] = png_compression_var.get()
                
                elif format_str == 'WEBP':
                    save_args['quality'] = webp_quality_var.get()
                    save_args['lossless'] = webp_lossless_var.get()
                    save_args['method'] = webp_method_var.get()
                
                elif format_str == 'TIFF':
                    save_args['compression'] = tiff_compression_var.get()
                
                elif format_str == 'GIF':
                    save_args['optimize'] = gif_optimize_var.get()
                
                # Save to BytesIO to estimate size
                save_img = preview_img.copy()
                
                # Convert to appropriate mode for the format
                if format_str == 'JPEG' or format_str == 'JPG':
                    if save_img.mode == 'RGBA':
                        # Create a white background
                        background = Image.new('RGB', save_img.size, (255, 255, 255))
                        background.paste(save_img, mask=save_img.split()[3])
                        save_img = background
                    elif save_img.mode != 'RGB':
                        save_img = save_img.convert('RGB')
                
                save_img.save(img_byte_arr, format=format_str, **save_args)
                estimated_size = len(img_byte_arr.getvalue())
                
                # Update info label
                info_label.config(text=f"Format: {format_str} | Mode: {mode_str} | "
                                      f"Dimensions: {img_width} × {img_height} pixels | "
                                      f"Estimated Size: {self._format_file_size(estimated_size)}")
                
            except Exception as e:
                # Show error in preview
                preview_canvas.delete("all")
                preview_canvas.create_text(
                    canvas_width // 2, canvas_height // 2,
                    text=f"Preview Error: {str(e)}",
                    fill="red", anchor=tk.CENTER
                )
                
                # Update info label
                info_label.config(text=f"Error: {str(e)}")
        
        # Info label for dimensions and file size
        info_label = ttk.Label(preview_frame, text="")
        info_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Update preview when canvas is configured
        def on_canvas_configure(event):
            update_preview()
        
        preview_canvas.bind("<Configure>", on_canvas_configure)
        
        # Update preview when options change
        for var in [format_var, color_mode_var, resize_var, width_var, height_var, 
                   aspect_var, resize_mode_var, resampling_var, optimize_var]:
            var.trace_add("write", lambda *args: update_preview())
        
        # Update preview when format-specific options change
        for var in [jpeg_quality_var, jpeg_progressive_var, jpeg_optimize_var, jpeg_subsampling_var,
                   png_compression_var, png_optimize_var, png_bits_var,
                   webp_quality_var, webp_lossless_var, webp_method_var,
                   tiff_compression_var, tiff_dpi_var,
                   gif_optimize_var, gif_transparency_var,
                   bg_color_var, grayscale_method_var, bw_threshold_var,
                   palette_colors_var, palette_dither_var]:
            var.trace_add("write", lambda *args: update_preview())
        
        # Output options
        output_frame = ttk.LabelFrame(main_frame, text="Output")
        output_frame.pack(fill=tk.X, pady=10)
        
        # Output directory
        output_dir_frame = ttk.Frame(output_frame)
        output_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_dir_frame, text="Save To:").pack(side=tk.LEFT)
        
        # Get last directory from controller if available
        last_dir = ""
        if self.controller and hasattr(self.controller, 'get_last_directory'):
            last_dir = self.controller.get_last_directory() or ""
        
        output_dir_var = tk.StringVar(value=last_dir)
        output_dir_entry = ttk.Entry(output_dir_frame, textvariable=output_dir_var, width=40)
        output_dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_output_dir():
            directory = filedialog.askdirectory(
                title="Select Output Directory",
                initialdir=output_dir_var.get() or os.path.expanduser('~')
            )
            if directory:
                output_dir_var.set(directory)
        
        ttk.Button(output_dir_frame, text="Browse...", command=browse_output_dir).pack(side=tk.LEFT)
        
        # Filename
        filename_frame = ttk.Frame(output_frame)
        filename_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filename_frame, text="Filename:").pack(side=tk.LEFT)
        
        # Generate default filename
        default_filename = "converted_image"
        if hasattr(image, 'filename'):
            base_name = os.path.splitext(os.path.basename(image.filename))[0]
            default_filename = f"{base_name}_converted"
        
        filename_var = tk.StringVar(value=default_filename)
        filename_entry = ttk.Entry(filename_frame, textvariable=filename_var, width=30)
        filename_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # File extension label (updates based on format)
        ext_label = ttk.Label(filename_frame, text=f".{format_var.get().lower()}")
        ext_label.pack(side=tk.LEFT)
        
        # Update extension label when format changes
        def update_ext_label(*args):
            format_str = format_var.get().lower()
            if format_str == 'jpeg':
                format_str = 'jpg'
            ext_label.config(text=f".{format_str}")
        
        format_var.trace_add("write", update_ext_label)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Convert button
        def on_convert():
            try:
                # Get output path
                output_dir = output_dir_var.get()
                if not output_dir:
                    messagebox.showerror("Error", "Please select an output directory.")
                    return
                
                # Create directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Get filename
                filename = filename_var.get()
                if not filename:
                    messagebox.showerror("Error", "Please enter a filename.")
                    return
                
                # Get format
                format_str = format_var.get()
                ext = format_str.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                # Full output path
                output_path = os.path.join(output_dir, f"{filename}.{ext}")
                
                # Check if file exists
                if os.path.exists(output_path):
                    overwrite = messagebox.askyesno(
                        "File Exists",
                        f"The file {filename}.{ext} already exists. Do you want to overwrite it?"
                    )
                    if not overwrite:
                        return
                
                # Create a copy of the image
                converted_img = image.copy()
                
                # Apply color mode conversion if needed
                mode = color_mode_var.get()
                if mode != converted_img.mode:
                    # Get color conversion options
                    color_options = {}
                    
                    if mode == 'RGB' and converted_img.mode == 'RGBA':
                        color_options['background_color'] = bg_color_var.get()
                    
                    if mode == 'L':
                        color_options['method'] = grayscale_method_var.get()
                    
                    if mode == '1':
                        color_options['threshold'] = bw_threshold_var.get()
                    
                    if mode == 'P':
                        color_options['colors'] = palette_colors_var.get()
                        color_options['dither'] = palette_dither_var.get()
                    
                    # Convert color mode
                    converted_img = self.convert_color_mode(converted_img, mode, **color_options)
                
                # Apply resize if enabled
                if resize_var.get():
                    new_width = width_var.get()
                    new_height = height_var.get()
                    
                    # Get resize options
                    resize_options = {
                        'maintain_aspect': aspect_var.get(),
                        'resize_mode': resize_mode_var.get(),
                        'resampling': resampling_var.get()
                    }
                    
                    # Resize the image
                    converted_img = self.resize_image(converted_img, new_width, new_height, **resize_options)
                
                # Get format-specific options
                format_options = {
                    'format': format_str,
                    'preserve_metadata': preserve_var.get()
                }
                
                if format_str == 'JPEG' or format_str == 'JPG':
                    format_options['quality'] = jpeg_quality_var.get()
                    format_options['progressive'] = jpeg_progressive_var.get()
                    format_options['optimize'] = jpeg_optimize_var.get()
                    format_options['subsampling'] = jpeg_subsampling_var.get()
                
                elif format_str == 'PNG':
                    format_options['compression'] = png_compression_var.get()
                    format_options['optimize'] = png_optimize_var.get()
                    format_options['bits'] = png_bits_var.get()
                
                elif format_str == 'WEBP':
                    format_options['quality'] = webp_quality_var.get()
                    format_options['lossless'] = webp_lossless_var.get()
                    format_options['method'] = webp_method_var.get()
                
                elif format_str == 'TIFF':
                    format_options['compression'] = tiff_compression_var.get()
                    format_options['resolution'] = tiff_dpi_var.get()
                
                elif format_str == 'GIF':
                    format_options['optimize'] = gif_optimize_var.get()
                    format_options['transparency'] = gif_transparency_var.get()
                
                elif format_str == 'PDF':
                    format_options['page_size'] = pdf_page_var.get()
                    format_options['resolution'] = pdf_dpi_var.get()
                
                # Apply optimization if enabled
                if optimize_var.get():
                    converted_img, opt_options = self.optimize_image(converted_img, format_str, **format_options)
                    # Merge optimization options with format options
                    format_options.update(opt_options)
                
                # Save the converted image
                result = self.save_converted_image(converted_img, output_path, **format_options)
                
                if result:
                    messagebox.showinfo("Conversion Complete", 
                                       f"Image converted successfully to {format_str} format.\n"
                                       f"Saved to: {output_path}")
                    
                    # Update settings
                    self.settings['default_format'] = format_str
                    if format_str == 'JPEG' or format_str == 'JPG':
                        self.settings['jpeg_quality'] = jpeg_quality_var.get()
                    elif format_str == 'PNG':
                        self.settings['png_compression'] = png_compression_var.get()
                    elif format_str == 'WEBP':
                        self.settings['webp_quality'] = webp_quality_var.get()
                        self.settings['webp_lossless'] = webp_lossless_var.get()
                    elif format_str == 'TIFF':
                        self.settings['tiff_compression'] = tiff_compression_var.get()
                    elif format_str == 'GIF':
                        self.settings['gif_optimize'] = gif_optimize_var.get()
                    
                    self.settings['preserve_metadata'] = preserve_var.get()
                    
                    # Close dialog
                    dialog.destroy()
                else:
                    messagebox.showerror("Conversion Failed", 
                                        "Failed to convert image. Please check the settings and try again.")
                
            except Exception as e:
                messagebox.showerror("Conversion Error", f"An error occurred during conversion:\n{str(e)}")
        
        ttk.Button(button_frame, text="Convert", command=on_convert).pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Initialize UI
        show_format_options()  # Show initial format options
        show_color_options()   # Show initial color mode options
        
        # Initialize preview
        dialog.update_idletasks()
        update_preview()
        
        # Center the dialog on the parent window
        dialog.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        return dialog
    
    def _format_file_size(self, size_bytes):
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"


def test_image_converter():
    """Test function for the image converter."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create a simple UI for testing
        root = tk.Tk()
        root.title("Image Converter Test")
        root.geometry("800x600")
        
        # Create the image converter
        converter = ImageConverter()
        
        # Current image
        current_image = [None]
        
        # Create a frame for controls
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Open image function
        def open_image():
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp")]
            )
            if file_path:
                try:
                    img = Image.open(file_path)
                    current_image[0] = img
                    update_preview(img)
                    status_var.set(f"Opened: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open image: {str(e)}")
        
        ttk.Button(control_frame, text="Open Image", command=open_image).pack(pady=5)
        
        # Convert image function
        def convert_image():
            if current_image[0] is None:
                messagebox.showinfo("Info", "Please open an image first.")
                return
            
            dialog = converter.create_converter_dialog(root, current_image[0])
            root.wait_window(dialog)
        
        ttk.Button(control_frame, text="Convert Image", command=convert_image).pack(pady=5)
        
        # Quick conversion functions
        quick_frame = ttk.LabelFrame(control_frame, text="Quick Convert")
        quick_frame.pack(fill=tk.X, pady=10)
        
        def quick_convert(format_str):
            if current_image[0] is None:
                messagebox.showinfo("Info", "Please open an image first.")
                return
            
            try:
                # Convert to the specified format
                converted = converter.convert_image(current_image[0], format_str)
                
                # Show save dialog
                file_path = filedialog.asksaveasfilename(
                    title=f"Save as {format_str}",
                    defaultextension=f".{format_str.lower()}",
                    filetypes=[(f"{format_str} files", f"*.{format_str.lower()}")]
                )
                
                if file_path:
                    # Save the converted image
                    result = converter.save_converted_image(converted, file_path, format=format_str)
                    
                    if result:
                        messagebox.showinfo("Success", f"Image converted and saved as {format_str}.")
                        status_var.set(f"Converted to {format_str}: {os.path.basename(file_path)}")
                    else:
                        messagebox.showerror("Error", f"Failed to save {format_str} image.")
            
            except Exception as e:
                messagebox.showerror("Error", f"Conversion error: {str(e)}")
        
        # Quick conversion buttons
        ttk.Button(quick_frame, text="To JPEG", 
                  command=lambda: quick_convert("JPEG")).pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_frame, text="To PNG", 
                  command=lambda: quick_convert("PNG")).pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_frame, text="To WebP", 
                  command=lambda: quick_convert("WEBP")).pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_frame, text="To GIF", 
                  command=lambda: quick_convert("GIF")).pack(fill=tk.X, pady=2)
        
        # Color mode conversion
        color_frame = ttk.LabelFrame(control_frame, text="Color Mode")
        color_frame.pack(fill=tk.X, pady=10)
        
        def convert_color_mode(mode):
            if current_image[0] is None:
                messagebox.showinfo("Info", "Please open an image first.")
                return
            
            try:
                # Convert to the specified color mode
                converted = converter.convert_color_mode(current_image[0], mode)
                
                # Update preview
                update_preview(converted)
                current_image[0] = converted
                status_var.set(f"Converted to {mode} color mode")
            
            except Exception as e:
                messagebox.showerror("Error", f"Color mode conversion error: {str(e)}")
        
        # Color mode buttons
        ttk.Button(color_frame, text="RGB", 
                  command=lambda: convert_color_mode("RGB")).pack(fill=tk.X, pady=2)
        
        ttk.Button(color_frame, text="RGBA", 
                  command=lambda: convert_color_mode("RGBA")).pack(fill=tk.X, pady=2)
        
        ttk.Button(color_frame, text="Grayscale", 
                  command=lambda: convert_color_mode("L")).pack(fill=tk.X, pady=2)
        
        ttk.Button(color_frame, text="Black & White", 
                  command=lambda: convert_color_mode("1")).pack(fill=tk.X, pady=2)
        
        # Create a preview area
        preview_frame = ttk.LabelFrame(root, text="Preview")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        preview_canvas = tk.Canvas(preview_frame, bg="#f0f0f0")
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
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
                preview_canvas_width = 600
                preview_canvas_height = 400
            
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
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_image_converter()