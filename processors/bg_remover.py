import os
import io
import time
import threading
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk

# Try to import rembg for automatic background removal
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# Try to import OpenCV for additional methods
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class BackgroundRemover:
    """
    A class for removing backgrounds from images using various methods.
    Supports automatic removal with rembg, color-based removal, and manual selection.
    """
    
    def __init__(self, controller=None):
        """
        Initialize the background remover.
        
        Args:
            controller: The controller object that handles the application logic
        """
        self.controller = controller
        self.progress_window = None
        self.progress_bar = None
        self.cancel_flag = False
        self.current_thread = None
        
        # Default settings
        self.settings = {
            'method': 'auto',  # 'auto', 'color', 'manual'
            'color_tolerance': 30,
            'edge_feather': 2,
            'model': 'u2net',  # For rembg
            'post_process': True
        }
    
    def remove_background(self, image, **kwargs):
        """
        Remove the background from an image.
        
        Args:
            image: PIL Image object
            **kwargs: Additional settings to override defaults
            
        Returns:
            PIL Image with transparent background
        """
        # Update settings with any provided kwargs
        settings = self.settings.copy()
        settings.update(kwargs)
        
        # Choose removal method
        method = settings['method']
        
        if method == 'auto':
            return self._auto_remove(image, settings)
        elif method == 'color':
            return self._color_based_remove(image, settings)
        elif method == 'manual':
            return self._manual_remove(image, settings)
        else:
            raise ValueError(f"Unknown background removal method: {method}")
    
    def remove_background_async(self, image, callback=None, **kwargs):
        """
        Remove the background asynchronously to avoid blocking the UI.
        
        Args:
            image: PIL Image object
            callback: Function to call with the result
            **kwargs: Additional settings to override defaults
        """
        # Cancel any existing operation
        self.cancel_operation()
        
        # Create and show progress window
        self._create_progress_window()
        
        # Reset cancel flag
        self.cancel_flag = False
        
        # Start background removal in a separate thread
        self.current_thread = threading.Thread(
            target=self._background_removal_thread,
            args=(image, callback, kwargs)
        )
        self.current_thread.daemon = True
        self.current_thread.start()
    
    def _background_removal_thread(self, image, callback, kwargs):
        """Thread function for background removal."""
        try:
            # Update progress
            self._update_progress(0, "Initializing...")
            
            # Process the image
            result = self.remove_background(image, **kwargs)
            
            # Check if operation was cancelled
            if self.cancel_flag:
                if callback:
                    callback(None)
                return
            
            # Update progress
            self._update_progress(100, "Complete")
            
            # Close progress window
            self._close_progress_window()
            
            # Call the callback with the result
            if callback:
                callback(result)
                
        except Exception as e:
            # Handle errors
            self._close_progress_window()
            if callback:
                callback(None, str(e))
    
    def _create_progress_window(self):
        """Create a progress window for background removal."""
        if self.progress_window is not None:
            return
            
        # Create a new toplevel window
        self.progress_window = tk.Toplevel()
        self.progress_window.title("Removing Background")
        self.progress_window.geometry("300x120")
        self.progress_window.resizable(False, False)
        
        # Make it modal
        self.progress_window.transient(self.controller.root if hasattr(self.controller, 'root') else None)
        self.progress_window.grab_set()
        
        # Center the window
        if hasattr(self.controller, 'root'):
            root = self.controller.root
            x = root.winfo_rootx() + (root.winfo_width() - 300) // 2
            y = root.winfo_rooty() + (root.winfo_height() - 120) // 2
            self.progress_window.geometry(f"+{x}+{y}")
        
        # Add a label
        self.progress_label = ttk.Label(self.progress_window, text="Removing background...")
        self.progress_label.pack(pady=(15, 5))
        
        # Add a progress bar
        self.progress_bar = ttk.Progressbar(self.progress_window, length=250, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Add a status label
        self.status_label = ttk.Label(self.progress_window, text="")
        self.status_label.pack(pady=5)
        
        # Add a cancel button
        cancel_button = ttk.Button(self.progress_window, text="Cancel", command=self.cancel_operation)
        cancel_button.pack(pady=5)
        
        # Handle window close
        self.progress_window.protocol("WM_DELETE_WINDOW", self.cancel_operation)
    
    def _update_progress(self, value, status=""):
        """Update the progress bar and status."""
        if self.progress_window and self.progress_bar:
            # Schedule the update on the main thread
            self.progress_window.after(0, self._do_update_progress, value, status)
    
    def _do_update_progress(self, value, status):
        """Actually update the progress UI (called on main thread)."""
        if self.progress_bar:
            self.progress_bar['value'] = value
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label['text'] = status
        
        # Update the window
        if self.progress_window:
            self.progress_window.update()
    
    def _close_progress_window(self):
        """Close the progress window."""
        if self.progress_window:
            # Schedule the close on the main thread
            self.progress_window.after(0, self._do_close_progress_window)
    
    def _do_close_progress_window(self):
        """Actually close the progress window (called on main thread)."""
        if self.progress_window:
            self.progress_window.grab_release()
            self.progress_window.destroy()
            self.progress_window = None
            self.progress_bar = None
    
    def cancel_operation(self):
        """Cancel the current background removal operation."""
        self.cancel_flag = True
        if self.current_thread and self.current_thread.is_alive():
            # We can't forcibly terminate the thread, but we can set a flag
            # that the thread should check periodically
            self.current_thread = None
        
        self._close_progress_window()
    
    def _auto_remove(self, image, settings):
        """
        Automatically remove background using rembg.
        
        Args:
            image: PIL Image object
            settings: Dictionary of settings
            
        Returns:
            PIL Image with transparent background
        """
        if not REMBG_AVAILABLE:
            raise ImportError(
                "The 'rembg' package is required for automatic background removal. "
                "Install it with: pip install rembg"
            )
        
        try:
            # Convert image to PNG format in memory
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Update progress
            self._update_progress(10, "Preparing image...")
            
            # Get model name
            model_name = settings.get('model', 'u2net')
            
            # Remove background
            self._update_progress(20, f"Removing background with {model_name} model...")
            
            # Process in chunks to provide progress updates
            result = None
            
            # We can't easily track progress with rembg, so we'll simulate it
            for i in range(3):
                if self.cancel_flag:
                    return None
                
                # Update progress
                self._update_progress(20 + i * 20, "Processing image...")
                time.sleep(0.1)  # Small delay to show progress
            
            # Actually remove the background
            output = rembg_remove(
                img_byte_arr.getvalue(),
                model_name=model_name,
                alpha_matting=settings.get('post_process', True),
                alpha_matting_foreground_threshold=settings.get('foreground_threshold', 240),
                alpha_matting_background_threshold=settings.get('background_threshold', 10),
                alpha_matting_erode_size=settings.get('erode_size', 10)
            )
            
            # Update progress
            self._update_progress(80, "Finalizing image...")
            
            # Convert back to PIL Image
            result = Image.open(io.BytesIO(output)).convert("RGBA")
            
            # Post-processing if enabled
            if settings.get('post_process', True):
                self._update_progress(90, "Post-processing...")
                result = self._post_process_image(result, settings)
            
            self._update_progress(100, "Complete")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error in automatic background removal: {str(e)}")
    
    def _color_based_remove(self, image, settings):
        """
        Remove background based on color similarity.
        
        Args:
            image: PIL Image object
            settings: Dictionary of settings
            
        Returns:
            PIL Image with transparent background
        """
        try:
            # Ensure image is RGBA
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Get the background color (default to corners)
            bg_color = settings.get('bg_color', None)
            
            if bg_color is None:
                # Sample from the corners
                width, height = image.size
                corners = [
                    image.getpixel((0, 0)),
                    image.getpixel((width-1, 0)),
                    image.getpixel((0, height-1)),
                    image.getpixel((width-1, height-1))
                ]
                
                # Use the most common corner color
                bg_color = max(set(corners), key=corners.count)
            
            # Get tolerance
            tolerance = settings.get('color_tolerance', 30)
            
            # Update progress
            self._update_progress(10, "Analyzing image...")
            
            # Convert to numpy array for faster processing
            img_array = np.array(image)
            
            # Create alpha mask
            alpha = np.ones((image.height, image.width), dtype=np.uint8) * 255
            
            # Update progress
            self._update_progress(20, "Creating transparency mask...")
            
            # Process in chunks for progress updates
            chunk_size = image.height // 10
            for i in range(0, image.height, chunk_size):
                if self.cancel_flag:
                    return None
                    
                end = min(i + chunk_size, image.height)
                
                # Calculate color distance
                r_diff = np.abs(img_array[i:end, :, 0] - bg_color[0])
                g_diff = np.abs(img_array[i:end, :, 1] - bg_color[1])
                b_diff = np.abs(img_array[i:end, :, 2] - bg_color[2])
                
                # Calculate total difference
                diff = np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)
                
                # Create mask based on tolerance
                mask = diff > tolerance
                
                # Apply to alpha channel
                alpha[i:end][~mask] = 0
                
                # Update progress
                progress = 20 + (i / image.height) * 60
                self._update_progress(progress, "Processing transparency...")
            
            # Apply feathering if requested
            feather = settings.get('edge_feather', 2)
            if feather > 0 and OPENCV_AVAILABLE:
                self._update_progress(80, "Feathering edges...")
                
                # Apply Gaussian blur to alpha channel
                alpha = cv2.GaussianBlur(alpha, (feather*2+1, feather*2+1), 0)
            
            # Update progress
            self._update_progress(90, "Finalizing image...")
            
            # Create new image with transparency
            result = Image.new('RGBA', image.size)
            result.paste(image, (0, 0), Image.fromarray(alpha))
            
            # Post-processing
            if settings.get('post_process', True):
                result = self._post_process_image(result, settings)
            
            self._update_progress(100, "Complete")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error in color-based background removal: {str(e)}")
    
    def _manual_remove(self, image, settings):
        """
        Remove background based on manual selection.
        This would typically be implemented in the UI layer.
        
        Args:
            image: PIL Image object
            settings: Dictionary of settings
            
        Returns:
            PIL Image with transparent background
        """
        # This would typically be implemented in the UI layer
        # Here we just provide a placeholder implementation
        
        # If we have a mask in the settings, use it
        mask = settings.get('mask', None)
        if mask is None:
            raise ValueError("Manual background removal requires a mask")
        
        # Ensure image is RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Apply the mask
        result = Image.new('RGBA', image.size)
        result.paste(image, (0, 0), mask)
        
        # Post-processing
        if settings.get('post_process', True):
            result = self._post_process_image(result, settings)
        
        return result
    
    def _post_process_image(self, image, settings):
        """
        Apply post-processing to improve the result.
        
        Args:
            image: PIL Image with transparent background
            settings: Dictionary of settings
            
        Returns:
            Improved PIL Image
        """
        # Ensure image is RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get alpha channel
        r, g, b, a = image.split()
        
        # Remove semi-transparent pixels at the edges if requested
        if settings.get('clean_edges', True) and OPENCV_AVAILABLE:
            # Convert alpha to numpy array
            alpha_np = np.array(a)
            
            # Create a binary mask of fully opaque pixels
            opaque_mask = alpha_np > 240
            
            # Apply morphological operations to clean edges
            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(opaque_mask.astype(np.uint8) * 255, 
                                           cv2.MORPH_OPEN, kernel)
            
            # Create new alpha channel
            new_alpha = Image.fromarray(cleaned_mask)
            
            # Merge back
            image = Image.merge('RGBA', (r, g, b, new_alpha))
        
        return image
    
    def show_settings_dialog(self, parent):
        """
        Show a dialog to configure background removal settings.
        
        Args:
            parent: Parent window
            
        Returns:
            Dictionary of updated settings or None if cancelled
        """
        # Create a new toplevel window
        dialog = tk.Toplevel(parent)
        dialog.title("Background Removal Settings")
        dialog.geometry("400x450")
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
        
        # Method selection
        ttk.Label(main_frame, text="Removal Method:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        method_var = tk.StringVar(value=self.settings['method'])
        
        method_frame = ttk.Frame(main_frame)
        method_frame.grid(row=0, column=1, sticky=tk.W, pady=(0, 5))
        
        ttk.Radiobutton(method_frame, text="Automatic (AI)", variable=method_var, 
                       value="auto").pack(anchor=tk.W)
        
        auto_available = ttk.Label(method_frame, 
                                  text="✓ Available" if REMBG_AVAILABLE else "✗ Not installed",
                                  foreground="green" if REMBG_AVAILABLE else "red")
        auto_available.pack(anchor=tk.W, padx=(20, 0))
        
        ttk.Radiobutton(method_frame, text="Color-based", variable=method_var, 
                       value="color").pack(anchor=tk.W, pady=(5, 0))
        ttk.Radiobutton(method_frame, text="Manual Selection", variable=method_var, 
                       value="manual").pack(anchor=tk.W, pady=(5, 0))
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Auto removal settings
        auto_frame = ttk.LabelFrame(main_frame, text="Automatic Removal Settings")
        auto_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        ttk.Label(auto_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        model_var = tk.StringVar(value=self.settings.get('model', 'u2net'))
        model_combo = ttk.Combobox(auto_frame, textvariable=model_var, width=15)
        model_combo['values'] = ('u2net', 'u2netp', 'u2net_human_seg', 'silueta')
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
                # Alpha matting options
        alpha_var = tk.BooleanVar(value=self.settings.get('post_process', True))
        ttk.Checkbutton(auto_frame, text="Use alpha matting", variable=alpha_var).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Foreground threshold
        ttk.Label(auto_frame, text="Foreground threshold:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        fg_threshold_var = tk.IntVar(value=self.settings.get('foreground_threshold', 240))
        fg_threshold_scale = ttk.Scale(auto_frame, from_=0, to=255, variable=fg_threshold_var, 
                                      orient=tk.HORIZONTAL)
        fg_threshold_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        fg_threshold_label = ttk.Label(auto_frame, text=str(fg_threshold_var.get()))
        fg_threshold_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_fg_label(*args):
            fg_threshold_label.config(text=str(fg_threshold_var.get()))
        
        fg_threshold_var.trace_add("write", update_fg_label)
        
        # Background threshold
        ttk.Label(auto_frame, text="Background threshold:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        bg_threshold_var = tk.IntVar(value=self.settings.get('background_threshold', 10))
        bg_threshold_scale = ttk.Scale(auto_frame, from_=0, to=255, variable=bg_threshold_var, 
                                      orient=tk.HORIZONTAL)
        bg_threshold_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        bg_threshold_label = ttk.Label(auto_frame, text=str(bg_threshold_var.get()))
        bg_threshold_label.grid(row=3, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_bg_label(*args):
            bg_threshold_label.config(text=str(bg_threshold_var.get()))
        
        bg_threshold_var.trace_add("write", update_bg_label)
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=3, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Color-based settings
        color_frame = ttk.LabelFrame(main_frame, text="Color-based Removal Settings")
        color_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Color tolerance
        ttk.Label(color_frame, text="Color tolerance:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        tolerance_var = tk.IntVar(value=self.settings.get('color_tolerance', 30))
        tolerance_scale = ttk.Scale(color_frame, from_=0, to=100, variable=tolerance_var, 
                                   orient=tk.HORIZONTAL)
        tolerance_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        tolerance_label = ttk.Label(color_frame, text=str(tolerance_var.get()))
        tolerance_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_tolerance_label(*args):
            tolerance_label.config(text=str(tolerance_var.get()))
        
        tolerance_var.trace_add("write", update_tolerance_label)
        
        # Edge feathering
        ttk.Label(color_frame, text="Edge feathering:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        feather_var = tk.IntVar(value=self.settings.get('edge_feather', 2))
        feather_scale = ttk.Scale(color_frame, from_=0, to=10, variable=feather_var, 
                                 orient=tk.HORIZONTAL)
        feather_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        feather_label = ttk.Label(color_frame, text=str(feather_var.get()))
        feather_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_feather_label(*args):
            feather_label.config(text=str(feather_var.get()))
        
        feather_var.trace_add("write", update_feather_label)
        
        # Background color picker
        ttk.Label(color_frame, text="Background color:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Color display
        color_preview = tk.Canvas(color_frame, width=30, height=20, bg="#ffffff")
        color_preview.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color picker button
        def pick_color():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose background color")
            if color[1]:
                color_preview.config(bg=color[1])
        
        ttk.Button(color_frame, text="Pick Color", command=pick_color).grid(
            row=2, column=2, padx=5, pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=5, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Post-processing settings
        post_frame = ttk.LabelFrame(main_frame, text="Post-processing Settings")
        post_frame.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Clean edges
        clean_edges_var = tk.BooleanVar(value=self.settings.get('clean_edges', True))
        ttk.Checkbutton(post_frame, text="Clean edges", variable=clean_edges_var).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Result variable
        result = [None]
        
        # OK button
        def on_ok():
            # Collect settings
            new_settings = {
                'method': method_var.get(),
                'model': model_var.get(),
                'post_process': alpha_var.get(),
                'foreground_threshold': fg_threshold_var.get(),
                'background_threshold': bg_threshold_var.get(),
                'color_tolerance': tolerance_var.get(),
                'edge_feather': feather_var.get(),
                'clean_edges': clean_edges_var.get(),
                'bg_color': color_preview.cget('bg')
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
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result[0]
    
    @staticmethod
    def is_available():
        """Check if automatic background removal is available."""
        return REMBG_AVAILABLE
    
    @staticmethod
    def install_instructions():
        """Get installation instructions for required packages."""
        instructions = []
        
        if not REMBG_AVAILABLE:
            instructions.append(
                "For automatic background removal, install rembg:\n"
                "pip install rembg"
            )
        
        if not OPENCV_AVAILABLE:
            instructions.append(
                "For advanced features, install OpenCV:\n"
                "pip install opencv-python"
            )
        
        return "\n\n".join(instructions) if instructions else None


class BackgroundGenerator:
    """
    A class for generating backgrounds for images with transparency.
    Supports solid colors, gradients, patterns, and textures.
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
            'type': 'solid',  # 'solid', 'gradient', 'pattern', 'texture'
            'color': '#ffffff',
            'gradient_color1': '#ffffff',
            'gradient_color2': '#000000',
            'gradient_direction': 'horizontal',
            'pattern': 'checkerboard',
            'texture_path': None
        }
    
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
        
        # Choose background type
        bg_type = settings['type']
        
        if bg_type == 'solid':
            return self._generate_solid_background(size, settings)
        elif bg_type == 'gradient':
            return self._generate_gradient_background(size, settings)
        elif bg_type == 'pattern':
            return self._generate_pattern_background(size, settings)
        elif bg_type == 'texture':
            return self._generate_texture_background(size, settings)
        else:
            raise ValueError(f"Unknown background type: {bg_type}")
    
    def _generate_solid_background(self, size, settings):
        """Generate a solid color background."""
        color = settings['color']
        return Image.new('RGBA', size, color)
    
    def _generate_gradient_background(self, size, settings):
        """Generate a gradient background."""
        from PIL import ImageDraw
        
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        direction = settings['gradient_direction']
        color1 = self._parse_color(settings['gradient_color1'])
        color2 = self._parse_color(settings['gradient_color2'])
        
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
            max_distance = width + height
            for y in range(height):
                for x in range(width):
                    # Calculate distance along diagonal
                    distance = (x + y) / max_distance
                    
                    # Calculate gradient color at this position
                    r = int(color1[0] + (color2[0] - color1[0]) * distance)
                    g = int(color1[1] + (color2[1] - color1[1]) * distance)
                    b = int(color1[2] + (color2[2] - color1[2]) * distance)
                    a = int(color1[3] + (color2[3] - color1[3]) * distance)
                    
                    draw.point((x, y), fill=(r, g, b, a))
                    
        elif direction == 'radial':
            # Calculate maximum distance from center
            center_x, center_y = width // 2, height // 2
            max_distance = (width**2 + height**2)**0.5 / 2
            
            for y in range(height):
                for x in range(width):
                    # Calculate distance from center
                    distance = ((x - center_x)**2 + (y - center_y)**2)**0.5 / max_distance
                    
                    # Calculate gradient color at this position
                    r = int(color1[0] + (color2[0] - color1[0]) * distance)
                    g = int(color1[1] + (color2[1] - color1[1]) * distance)
                    b = int(color1[2] + (color2[2] - color1[2]) * distance)
                    a = int(color1[3] + (color2[3] - color1[3]) * distance)
                    
                    draw.point((x, y), fill=(r, g, b, a))
        
        return image
    
    def _generate_pattern_background(self, size, settings):
        """Generate a pattern background."""
        pattern_type = settings.get('pattern', 'checkerboard')
        
        if pattern_type == 'checkerboard':
            return self._generate_checkerboard(size, settings)
        elif pattern_type == 'stripes':
            return self._generate_stripes(size, settings)
        elif pattern_type == 'dots':
            return self._generate_dots(size, settings)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    def _generate_checkerboard(self, size, settings):
        """Generate a checkerboard pattern."""
        from PIL import ImageDraw
        
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
        from PIL import ImageDraw
        
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
        from PIL import ImageDraw
        
        width, height = size
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        bg_color = self._parse_color(settings.get('pattern_bg_color', '#ffffff'))
        dot_color = self._parse_color(settings.get('pattern_dot_color', '#cccccc'))
        
        # Fill background
        draw.rectangle([0, 0, width, height], fill=bg_color)
        
        dot_size = settings.get('pattern_dot_size', 10)
        spacing = settings.get('pattern_spacing', 30)
        
        for y in range(spacing // 2, height, spacing):
            for x in range(spacing // 2, width, spacing):
                draw.ellipse([x - dot_size // 2, y - dot_size // 2, 
                             x + dot_size // 2, y + dot_size // 2], fill=dot_color)
        
        return image
    
    def _generate_texture_background(self, size, settings):
        """Generate a textured background from an image."""
        texture_path = settings.get('texture_path')
        
        if not texture_path or not os.path.exists(texture_path):
            # Fall back to solid color
            return self._generate_solid_background(size, settings)
        
        try:
            # Load texture image
            texture = Image.open(texture_path).convert('RGBA')
            
            # Resize to fill the target size
            texture = self._resize_to_fill(texture, size)
            
            # Crop to exact size
            left = (texture.width - size[0]) // 2
            top = (texture.height - size[1]) // 2
            texture = texture.crop((left, top, left + size[0], top + size[1]))
            
            return texture
            
        except Exception:
            # Fall back to solid color on error
            return self._generate_solid_background(size, settings)
    
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
    
    def _parse_color(self, color):
        """Parse a color string into RGBA tuple."""
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
        dialog.geometry("400x500")
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
        
        # Background type selection
        ttk.Label(main_frame, text="Background Type:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        bg_type_var = tk.StringVar(value=self.settings['type'])
        
        ttk.Radiobutton(main_frame, text="Solid Color", variable=bg_type_var, 
                       value="solid").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Gradient", variable=bg_type_var, 
                       value="gradient").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Pattern", variable=bg_type_var, 
                       value="pattern").grid(row=2, column=1, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Texture", variable=bg_type_var, 
                       value="texture").grid(row=3, column=1, sticky=tk.W)
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=4, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Create notebook for different background type settings
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=5, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        
        # Solid color settings
        solid_frame = ttk.Frame(notebook, padding=10)
        notebook.add(solid_frame, text="Solid Color")
        
        ttk.Label(solid_frame, text="Color:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Color display
        solid_color_preview = tk.Canvas(solid_frame, width=30, height=20, 
                                       bg=self.settings['color'])
        solid_color_preview.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Color picker button
        def pick_solid_color():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose background color", 
                                         initialcolor=self.settings['color'])
            if color[1]:
                solid_color_preview.config(bg=color[1])
        
        ttk.Button(solid_frame, text="Pick Color", command=pick_solid_color).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Gradient settings
        gradient_frame = ttk.Frame(notebook, padding=10)
        notebook.add(gradient_frame, text="Gradient")
        
        ttk.Label(gradient_frame, text="Start Color:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Start color display
        gradient_color1_preview = tk.Canvas(gradient_frame, width=30, height=20, 
                                          bg=self.settings['gradient_color1'])
        gradient_color1_preview.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Start color picker button
        def pick_gradient_color1():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose start color", 
                                         initialcolor=self.settings['gradient_color1'])
            if color[1]:
                gradient_color1_preview.config(bg=color[1])
        
        ttk.Button(gradient_frame, text="Pick Color", command=pick_gradient_color1).grid(
            row=0, column=2, padx=5, pady=5)
        
        ttk.Label(gradient_frame, text="End Color:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # End color display
        gradient_color2_preview = tk.Canvas(gradient_frame, width=30, height=20, 
                                          bg=self.settings['gradient_color2'])
        gradient_color2_preview.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # End color picker button
        def pick_gradient_color2():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose end color", 
                                         initialcolor=self.settings['gradient_color2'])
            if color[1]:
                gradient_color2_preview.config(bg=color[1])
        
        ttk.Button(gradient_frame, text="Pick Color", command=pick_gradient_color2).grid(
            row=1, column=2, padx=5, pady=5)
        
        # Direction
        ttk.Label(gradient_frame, text="Direction:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        direction_var = tk.StringVar(value=self.settings['gradient_direction'])
        direction_combo = ttk.Combobox(gradient_frame, textvariable=direction_var, width=15)
        direction_combo['values'] = ('horizontal', 'vertical', 'diagonal', 'radial')
        direction_combo.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Pattern settings
        pattern_frame = ttk.Frame(notebook, padding=10)
        notebook.add(pattern_frame, text="Pattern")
        
        ttk.Label(pattern_frame, text="Pattern Type:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        pattern_var = tk.StringVar(value=self.settings.get('pattern', 'checkerboard'))
        pattern_combo = ttk.Combobox(pattern_frame, textvariable=pattern_var, width=15)
        pattern_combo['values'] = ('checkerboard', 'stripes', 'dots')
        pattern_combo.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(pattern_frame, text="Color 1:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Pattern color 1 display
        pattern_color1_preview = tk.Canvas(pattern_frame, width=30, height=20, 
                                         bg=self.settings.get('pattern_color1', '#ffffff'))
        pattern_color1_preview.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Pattern color 1 picker button
        def pick_pattern_color1():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose pattern color 1", 
                                         initialcolor=self.settings.get('pattern_color1', '#ffffff'))
            if color[1]:
                pattern_color1_preview.config(bg=color[1])
        
        ttk.Button(pattern_frame, text="Pick Color", command=pick_pattern_color1).grid(
            row=1, column=2, padx=5, pady=5)
        
        ttk.Label(pattern_frame, text="Color 2:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Pattern color 2 display
        pattern_color2_preview = tk.Canvas(pattern_frame, width=30, height=20, 
                                         bg=self.settings.get('pattern_color2', '#cccccc'))
        pattern_color2_preview.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Pattern color 2 picker button
        def pick_pattern_color2():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose pattern color 2", 
                                         initialcolor=self.settings.get('pattern_color2', '#cccccc'))
            if color[1]:
                pattern_color2_preview.config(bg=color[1])
        
        ttk.Button(pattern_frame, text="Pick Color", command=pick_pattern_color2).grid(
            row=2, column=2, padx=5, pady=5)
        
        # Pattern size
        ttk.Label(pattern_frame, text="Size:").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        pattern_size_var = tk.IntVar(value=self.settings.get('pattern_size', 20))
        pattern_size_scale = ttk.Scale(pattern_frame, from_=5, to=50, variable=pattern_size_var, 
                                      orient=tk.HORIZONTAL)
        pattern_size_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        pattern_size_label = ttk.Label(pattern_frame, text=str(pattern_size_var.get()))
        pattern_size_label.grid(row=3, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_pattern_size_label(*args):
            pattern_size_label.config(text=str(pattern_size_var.get()))
        
        pattern_size_var.trace_add("write", update_pattern_size_label)
        
        # Texture settings
        texture_frame = ttk.Frame(notebook, padding=10)
        notebook.add(texture_frame, text="Texture")
        
        ttk.Label(texture_frame, text="Texture Image:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        texture_path_var = tk.StringVar(value=self.settings.get('texture_path', ''))
        texture_path_entry = ttk.Entry(texture_frame, textvariable=texture_path_var, width=25)
        texture_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Browse button
        def browse_texture():
            from tkinter import filedialog
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
        
        ttk.Button(texture_frame, text="Browse...", command=browse_texture).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Texture preview
        texture_preview_frame = ttk.LabelFrame(texture_frame, text="Preview")
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
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Result variable
        result = [None]
        
        # OK button
        def on_ok():
            # Collect settings
            new_settings = {
                'type': bg_type_var.get(),
                'color': solid_color_preview.cget('bg'),
                'gradient_color1': gradient_color1_preview.cget('bg'),
                'gradient_color2': gradient_color2_preview.cget('bg'),
                'gradient_direction': direction_var.get(),
                'pattern': pattern_var.get(),
                'pattern_color1': pattern_color1_preview.cget('bg'),
                'pattern_color2': pattern_color2_preview.cget('bg'),
                'pattern_size': pattern_size_var.get(),
                'texture_path': texture_path_var.get()
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
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result[0]


def test_background_remover():
    """Test function for the background remover."""
    try:
        from PIL import Image
        import tkinter as tk
        
        # Create a simple test image
        test_image = Image.new('RGB', (300, 200), color='white')
        
        # Create a simple UI for testing
        root = tk.Tk()
        root.title("Background Remover Test")
        root.geometry("400x300")
        
        # Create the background remover
        remover = BackgroundRemover()
        
        # Create a button to test settings dialog
        def show_settings():
            remover.show_settings_dialog(root)
        
        ttk.Button(root, text="Show Settings", command=show_settings).pack(pady=20)
        
        # Create a button to test background removal
        def remove_bg():
            result = remover.remove_background(test_image)
            if result:
                result.show()
        
        ttk.Button(root, text="Remove Background", command=remove_bg).pack(pady=20)
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Test failed: {e}")


def test_background_generator():
    """Test function for the background generator."""
    try:
        from PIL import Image
        import tkinter as tk
        
        # Create a simple UI for testing
        root = tk.Tk()
        root.title("Background Generator Test")
        root.geometry("400x300")
        
        # Create the background generator
        generator = BackgroundGenerator()
        
        # Create a button to test settings dialog
        def show_settings():
            generator.show_settings_dialog(root)
        
        ttk.Button(root, text="Show Settings", command=show_settings).pack(pady=20)
        
        # Create a button to test background generation
        def generate_bg():
            result = generator.generate_background((500, 300))
            if result:
                result.show()
        
        ttk.Button(root, text="Generate Background", command=generate_bg).pack(pady=20)
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_background_remover()
    test_background_generator()