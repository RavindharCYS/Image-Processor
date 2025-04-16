import os
import math
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk, Canvas, Frame, StringVar, IntVar, DoubleVar, BooleanVar

# Try to import OpenCV for additional features
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class ImageCropper:
    """
    A class for cropping images with various options including aspect ratio presets,
    rotation, and straightening.
    """
    
    def __init__(self, controller=None):
        """
        Initialize the image cropper.
        
        Args:
            controller: The controller object that handles the application logic
        """
        self.controller = controller
        
        # Default settings
        self.settings = {
            'aspect_ratio': 'free',  # free, original, 1:1, 16:9, 4:3, 3:2, etc.
            'custom_ratio_width': 1,
            'custom_ratio_height': 1,
            'rotation': 0,  # degrees
            'flip_horizontal': False,
            'flip_vertical': False,
            'auto_straighten': False,
            'grid_overlay': 'rule_of_thirds',  # none, rule_of_thirds, grid, golden_ratio
            'constrain_to_image': True
        }
        
        # Aspect ratio presets
        self.aspect_ratio_presets = {
            'free': (0, 0),  # No constraint
            'original': (-1, -1),  # Original image ratio
            '1:1': (1, 1),  # Square
            '16:9': (16, 9),  # Widescreen
            '4:3': (4, 3),  # Standard
            '3:2': (3, 2),  # Classic photo
            '5:4': (5, 4),  # Large format
            '3:1': (3, 1),  # Panorama
            '2:3': (2, 3),  # Portrait
            '9:16': (9, 16)  # Mobile
        }
    
    def crop_image(self, image, crop_rect, **kwargs):
        """
        Crop an image based on the given rectangle.
        
        Args:
            image: PIL Image object
            crop_rect: Tuple of (left, top, right, bottom) in pixels
            **kwargs: Additional settings to override defaults
            
        Returns:
            Cropped PIL Image
        """
        # Update settings with any provided kwargs
        settings = self.settings.copy()
        settings.update(kwargs)
        
        # Make a copy of the image to avoid modifying the original
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Apply rotation if needed
        rotation = settings.get('rotation', 0)
        if rotation != 0:
            # Rotate the image
            image = self._rotate_image(image, rotation)
            
            # Adjust crop rectangle for rotation
            # This is complex and would require recalculating the crop area
            # For simplicity, we'll assume the crop_rect is already adjusted for rotation
        
        # Apply flips if needed
        if settings.get('flip_horizontal', False):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if settings.get('flip_vertical', False):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Ensure crop rectangle is valid
        left, top, right, bottom = crop_rect
        left = max(0, min(left, image.width - 1))
        top = max(0, min(top, image.height - 1))
        right = max(left + 1, min(right, image.width))
        bottom = max(top + 1, min(bottom, image.height))
        
        # Crop the image
        cropped = image.crop((left, top, right, bottom))
        
        return cropped
    
    def _rotate_image(self, image, angle):
        """
        Rotate an image by the given angle.
        
        Args:
            image: PIL Image object
            angle: Rotation angle in degrees
            
        Returns:
            Rotated PIL Image
        """
        # For small angles, use a higher quality method
        if abs(angle) < 10 and OPENCV_AVAILABLE:
            # Convert to OpenCV format
            img_array = np.array(image)
            
            # Get rotation matrix
            height, width = img_array.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            
            # Calculate new dimensions
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)
            
            # Adjust rotation matrix
            rotation_matrix[0, 2] += new_width / 2 - center[0]
            rotation_matrix[1, 2] += new_height / 2 - center[1]
            
            # Perform rotation
            rotated = cv2.warpAffine(img_array, rotation_matrix, (new_width, new_height),
                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))
            
            # Convert back to PIL
            return Image.fromarray(rotated)
        else:
            # Use PIL's rotation
            # Expand to fit the entire rotated image
            return image.rotate(-angle, resample=Image.BICUBIC, expand=True)
    
    def auto_straighten(self, image):
        """
        Automatically straighten an image by detecting horizontal/vertical lines.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (straightened image, rotation angle)
        """
        if not OPENCV_AVAILABLE:
            return image, 0
        
        try:
            # Convert to OpenCV format
            img_array = np.array(image)
            
            # Convert to grayscale
            if img_array.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) == 0:
                return image, 0
            
            # Calculate the most common angle
            angles = []
            for line in lines:
                rho, theta = line[0]
                # Convert to degrees and normalize to -45 to 45
                angle = (theta * 180 / np.pi) % 180
                if angle > 90:
                    angle -= 180
                elif angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                
                # Only consider angles close to horizontal or vertical
                if abs(angle) < 45:
                    angles.append(angle)
            
            if not angles:
                return image, 0
            
            # Find the median angle
            median_angle = np.median(angles)
            
            # Only straighten if the angle is significant but not too large
            if abs(median_angle) < 0.5 or abs(median_angle) > 10:
                return image, 0
            
            # Rotate the image
            rotated = self._rotate_image(image, median_angle)
            
            return rotated, median_angle
            
        except Exception as e:
            print(f"Auto-straighten error: {e}")
            return image, 0
    
    def get_aspect_ratio(self, preset, original_width, original_height):
        """
        Get the width and height ratio for a given aspect ratio preset.
        
        Args:
            preset: Aspect ratio preset name
            original_width: Original image width
            original_height: Original image height
            
        Returns:
            Tuple of (width_ratio, height_ratio)
        """
        if preset == 'free':
            return (0, 0)  # No constraint
        
        if preset == 'original':
            return (original_width, original_height)
        
        if preset == 'custom':
            return (self.settings['custom_ratio_width'], self.settings['custom_ratio_height'])
        
        return self.aspect_ratio_presets.get(preset, (0, 0))
    
    def calculate_crop_rect(self, image_width, image_height, center_x, center_y, 
                           crop_width, crop_height, constrain=True):
        """
        Calculate the crop rectangle based on center point and dimensions.
        
        Args:
            image_width: Width of the original image
            image_height: Height of the original image
            center_x: X coordinate of the crop center
            center_y: Y coordinate of the crop center
            crop_width: Width of the crop area
            crop_height: Height of the crop area
            constrain: Whether to constrain the crop to the image bounds
            
        Returns:
            Tuple of (left, top, right, bottom)
        """
        left = center_x - crop_width / 2
        top = center_y - crop_height / 2
        right = center_x + crop_width / 2
        bottom = center_y + crop_height / 2
        
        if constrain:
            # Ensure the crop is within the image bounds
            if left < 0:
                right -= left
                left = 0
            if top < 0:
                bottom -= top
                top = 0
            if right > image_width:
                left -= (right - image_width)
                right = image_width
            if bottom > image_height:
                top -= (bottom - image_height)
                bottom = image_height
            
            # Ensure minimum size
            if right - left < 10:
                right = left + 10
            if bottom - top < 10:
                bottom = top + 10
            
            # Final bounds check
            left = max(0, min(left, image_width - 10))
            top = max(0, min(top, image_height - 10))
            right = max(left + 10, min(right, image_width))
            bottom = max(top + 10, min(bottom, image_height))
        
        return (int(left), int(top), int(right), int(bottom))
    
    def adjust_crop_for_aspect_ratio(self, image_width, image_height, crop_rect, aspect_ratio):
        """
        Adjust a crop rectangle to match the specified aspect ratio.
        
        Args:
            image_width: Width of the original image
            image_height: Height of the original image
            crop_rect: Current crop rectangle (left, top, right, bottom)
            aspect_ratio: Tuple of (width_ratio, height_ratio)
            
        Returns:
            Adjusted crop rectangle (left, top, right, bottom)
        """
        width_ratio, height_ratio = aspect_ratio
        
        # If no constraint, return the original
        if width_ratio == 0 or height_ratio == 0:
            return crop_rect
        
        left, top, right, bottom = crop_rect
        current_width = right - left
        current_height = bottom - top
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        
        # Calculate target ratio
        target_ratio = width_ratio / height_ratio
        current_ratio = current_width / current_height
        
        if abs(current_ratio - target_ratio) < 0.01:
            # Already close enough to the target ratio
            return crop_rect
        
        # Adjust dimensions to match the target ratio
        if current_ratio > target_ratio:
            # Too wide, adjust width
            new_width = current_height * target_ratio
            new_height = current_height
        else:
            # Too tall, adjust height
            new_width = current_width
            new_height = current_width / target_ratio
        
        # Calculate new crop rectangle
        return self.calculate_crop_rect(
            image_width, image_height, center_x, center_y, new_width, new_height
        )
    
    def draw_grid_overlay(self, image, crop_rect, grid_type='rule_of_thirds'):
        """
        Draw a grid overlay on the image for composition guidance.
        
        Args:
            image: PIL Image object
            crop_rect: Crop rectangle (left, top, right, bottom)
            grid_type: Type of grid to draw
            
        Returns:
            PIL Image with grid overlay
        """
        # Create a copy of the image
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        
        left, top, right, bottom = crop_rect
        width = right - left
        height = bottom - top
        
        # Draw crop rectangle
        draw.rectangle(crop_rect, outline=(255, 255, 255, 200), width=2)
        
        if grid_type == 'none':
            return overlay
        
        # Semi-transparent white lines
        line_color = (255, 255, 255, 128)
        
        if grid_type == 'rule_of_thirds':
            # Draw rule of thirds grid
            # Vertical lines
            for i in range(1, 3):
                x = left + width * i / 3
                draw.line([(x, top), (x, bottom)], fill=line_color, width=1)
            
            # Horizontal lines
            for i in range(1, 3):
                y = top + height * i / 3
                draw.line([(left, y), (right, y)], fill=line_color, width=1)
                
        elif grid_type == 'grid':
            # Draw a regular grid (5x5)
            # Vertical lines
            for i in range(1, 5):
                x = left + width * i / 5
                draw.line([(x, top), (x, bottom)], fill=line_color, width=1)
            
            # Horizontal lines
            for i in range(1, 5):
                y = top + height * i / 5
                draw.line([(left, y), (right, y)], fill=line_color, width=1)
                
        elif grid_type == 'golden_ratio':
            # Draw golden ratio grid (phi ≈ 1.618)
            phi = 1.618
            
            # Calculate golden sections
            golden_x1 = left + width / phi
            golden_x2 = right - width / phi
            golden_y1 = top + height / phi
            golden_y2 = bottom - height / phi
            
            # Vertical lines
            draw.line([(golden_x1, top), (golden_x1, bottom)], fill=line_color, width=1)
            draw.line([(golden_x2, top), (golden_x2, bottom)], fill=line_color, width=1)
            
            # Horizontal lines
            draw.line([(left, golden_y1), (right, golden_y1)], fill=line_color, width=1)
            draw.line([(left, golden_y2), (right, golden_y2)], fill=line_color, width=1)
            
        elif grid_type == 'diagonal':
            # Draw diagonal guides
            # Main diagonals
            draw.line([(left, top), (right, bottom)], fill=line_color, width=1)
            draw.line([(left, bottom), (right, top)], fill=line_color, width=1)
            
            # Center lines
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            draw.line([(center_x, top), (center_x, bottom)], fill=line_color, width=1)
            draw.line([(left, center_y), (right, center_y)], fill=line_color, width=1)
        
        return overlay
    
    def show_crop_dialog(self, parent, image):
        """
        Show a dialog for interactive cropping.
        
        Args:
            parent: Parent window
            image: PIL Image to crop
            
        Returns:
            Cropped PIL Image or None if cancelled
        """
        # Create a new toplevel window
        dialog = tk.Toplevel(parent)
        dialog.title("Crop Image")
        dialog.geometry("900x700")
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
        
        # Create a frame for the image canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create canvas for image display
        canvas = Canvas(canvas_frame, bg="#1e1e1e", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Create a frame for controls
        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)  # Prevent shrinking
        
        # Create controls
        ttk.Label(control_frame, text="Aspect Ratio:").pack(anchor=tk.W, pady=(0, 5))
        
        aspect_ratio_var = StringVar(value=self.settings['aspect_ratio'])
        aspect_ratio_combo = ttk.Combobox(control_frame, textvariable=aspect_ratio_var, width=15)
        aspect_ratio_combo['values'] = list(self.aspect_ratio_presets.keys()) + ['custom']
        aspect_ratio_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Custom ratio frame
        custom_ratio_frame = ttk.Frame(control_frame)
        custom_ratio_frame.pack(fill=tk.X, pady=(0, 10))
        
        custom_width_var = IntVar(value=self.settings['custom_ratio_width'])
        custom_width_spinbox = ttk.Spinbox(custom_ratio_frame, from_=1, to=100, 
                                          textvariable=custom_width_var, width=5)
        custom_width_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(custom_ratio_frame, text=":").pack(side=tk.LEFT, padx=5)
        
        custom_height_var = IntVar(value=self.settings['custom_ratio_height'])
        custom_height_spinbox = ttk.Spinbox(custom_ratio_frame, from_=1, to=100, 
                                           textvariable=custom_height_var, width=5)
        custom_height_spinbox.pack(side=tk.LEFT)
        
        # Show/hide custom ratio controls based on selection
        def update_custom_ratio_visibility(*args):
            if aspect_ratio_var.get() == 'custom':
                custom_ratio_frame.pack(fill=tk.X, pady=(0, 10))
            else:
                custom_ratio_frame.pack_forget()
        
        aspect_ratio_var.trace_add("write", update_custom_ratio_visibility)
        update_custom_ratio_visibility()  # Initial visibility
        
        # Rotation control
        ttk.Label(control_frame, text="Rotation:").pack(anchor=tk.W, pady=(10, 5))
        
        rotation_var = DoubleVar(value=self.settings['rotation'])
        rotation_scale = ttk.Scale(control_frame, from_=-45, to=45, variable=rotation_var, 
                                  orient=tk.HORIZONTAL)
        rotation_scale.pack(fill=tk.X)
        
        rotation_frame = ttk.Frame(control_frame)
        rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(rotation_frame, textvariable=rotation_var).pack(side=tk.LEFT)
        ttk.Label(rotation_frame, text="°").pack(side=tk.LEFT)
        
        # Reset rotation button
        ttk.Button(rotation_frame, text="Reset", command=lambda: rotation_var.set(0), 
                  width=8).pack(side=tk.RIGHT)
        
        # Flip controls
        flip_frame = ttk.Frame(control_frame)
        flip_frame.pack(fill=tk.X, pady=(0, 10))
        
        flip_h_var = BooleanVar(value=self.settings['flip_horizontal'])
        ttk.Checkbutton(flip_frame, text="Flip Horizontal", variable=flip_h_var).pack(
            side=tk.LEFT, padx=(0, 10))
        
        flip_v_var = BooleanVar(value=self.settings['flip_vertical'])
        ttk.Checkbutton(flip_frame, text="Flip Vertical", variable=flip_v_var).pack(side=tk.LEFT)
        
        # Auto-straighten
        auto_straighten_var = BooleanVar(value=self.settings['auto_straighten'])
        ttk.Checkbutton(control_frame, text="Auto-straighten", 
                       variable=auto_straighten_var).pack(anchor=tk.W, pady=(0, 10))
        
        # Grid overlay
        ttk.Label(control_frame, text="Grid Overlay:").pack(anchor=tk.W, pady=(10, 5))
        
        grid_var = StringVar(value=self.settings['grid_overlay'])
        grid_combo = ttk.Combobox(control_frame, textvariable=grid_var, width=15)
        grid_combo['values'] = ['none', 'rule_of_thirds', 'grid', 'golden_ratio', 'diagonal']
        grid_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Constrain to image
        constrain_var = BooleanVar(value=self.settings['constrain_to_image'])
        ttk.Checkbutton(control_frame, text="Constrain to Image", 
                       variable=constrain_var).pack(anchor=tk.W, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, text="Apply", command=lambda: on_apply()).pack(
            side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        
        # Reset button
        ttk.Button(control_frame, text="Reset Crop", command=lambda: reset_crop()).pack(
            fill=tk.X, pady=(10, 0))
        
        # Variables for tracking crop state
        crop_rect = [0, 0, 0, 0]  # left, top, right, bottom
        drag_start_x = 0
        drag_start_y = 0
        dragging = False
        drag_mode = None  # 'move', 'resize_nw', 'resize_ne', etc.
        
        # Image display variables
        image_id = None
        display_image = None
        scale_factor = 1.0
        
        # Result variable
        result = [None]
        
        # Function to apply crop and close dialog
        def on_apply():
            if image is None:
                dialog.destroy()
                return
            
            # Get final crop rectangle
            left, top, right, bottom = crop_rect
            
            # Scale back to original image coordinates
            left = int(left / scale_factor)
            top = int(top / scale_factor)
            right = int(right / scale_factor)
            bottom = int(bottom / scale_factor)
            
            # Apply rotation and flips first if needed
            processed_image = image.copy()
            
            # Apply flips
            if flip_h_var.get():
                processed_image = processed_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            if flip_v_var.get():
                processed_image = processed_image.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Apply rotation
            rotation = rotation_var.get()
            if rotation != 0:
                processed_image = self._rotate_image(processed_image, rotation)
                
                # Adjust crop rectangle for rotation
                # This is complex, so we'll use a simplified approach
                # We'll crop from the rotated image directly
                orig_width, orig_height = image.size
                new_width, new_height = processed_image.size
                
                # Calculate the center offset
                center_offset_x = (new_width - orig_width) / 2
                center_offset_y = (new_height - orig_height) / 2
                
                # Adjust crop coordinates
                left += center_offset_x
                right += center_offset_x
                top += center_offset_y
                bottom += center_offset_y
            
            # Ensure crop rectangle is valid
            left = max(0, min(left, processed_image.width - 1))
            top = max(0, min(top, processed_image.height - 1))
            right = max(left + 1, min(right, processed_image.width))
            bottom = max(top + 1, min(bottom, processed_image.height))
            
            # Crop the image
            cropped = processed_image.crop((left, top, right, bottom))
            
            # Set result and close dialog
            result[0] = cropped
            dialog.destroy()
        
        # Function to reset crop to full image
        def reset_crop():
            # Reset to full image
            if display_image:
                width, height = display_image.size
                crop_rect[0] = 0
                crop_rect[1] = 0
                crop_rect[2] = width
                crop_rect[3] = height
                update_crop_display()
        
        # Function to display the image with crop overlay
        def display_image_with_crop():
            if image is None:
                return
            
            # Calculate scale factor to fit the image in the canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet properly sized, use default size
                canvas_width = 640
                canvas_height = 480
            
            # Calculate scale factor
            img_width, img_height = image.size
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            
            # Use the smaller ratio to ensure the entire image fits
            nonlocal scale_factor
            scale_factor = min(width_ratio, height_ratio) * 0.9  # 90% to leave some margin
            
            # Resize image for display
            display_width = int(img_width * scale_factor)
            display_height = int(img_height * scale_factor)
            
            # Create a copy of the image for display
            nonlocal display_image
            display_image = image.copy()
            display_image = display_image.resize((display_width, display_height), Image.LANCZOS)
            
            # Initialize crop rectangle to full image if not set
            if crop_rect[2] == 0 or crop_rect[3] == 0:
                crop_rect[0] = 0
                crop_rect[1] = 0
                crop_rect[2] = display_width
                crop_rect[3] = display_height
            
            # Update the display
            update_crop_display()
            
            # Configure canvas scrolling
            canvas.config(scrollregion=(0, 0, display_width, display_height))
        
        # Function to update the crop display
        def update_crop_display():
            if display_image is None:
                return
            
            # Apply aspect ratio constraint if needed
            aspect_ratio_name = aspect_ratio_var.get()
            if aspect_ratio_name != 'free':
                # Get aspect ratio values
                if aspect_ratio_name == 'custom':
                    width_ratio = custom_width_var.get()
                    height_ratio = custom_height_var.get()
                elif aspect_ratio_name == 'original':
                    width_ratio, height_ratio = image.size
                else:
                    width_ratio, height_ratio = self.aspect_ratio_presets.get(aspect_ratio_name, (0, 0))
                
                # Adjust crop rectangle to match aspect ratio
                if width_ratio > 0 and height_ratio > 0:
                    display_width, display_height = display_image.size
                    crop_rect[:] = self.adjust_crop_for_aspect_ratio(
                        display_width, display_height, crop_rect, (width_ratio, height_ratio)
                    )
            
            # Draw grid overlay
            grid_type = grid_var.get()
            overlay = self.draw_grid_overlay(display_image, crop_rect, grid_type)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(overlay)
            
            # Update canvas
            nonlocal image_id
            if image_id:
                canvas.delete(image_id)
            
            image_id = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # Keep a reference to prevent garbage collection
            
            # Ensure crop rectangle is visible
            canvas.xview_moveto(max(0, (crop_rect[0] - 50) / display_image.width))
            canvas.yview_moveto(max(0, (crop_rect[1] - 50) / display_image.height))
        
        # Mouse event handlers for crop interaction
        def on_mouse_down(event):
            nonlocal dragging, drag_start_x, drag_start_y, drag_mode
            
            # Convert canvas coordinates to image coordinates
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            
            # Check if click is on a handle or inside the crop rectangle
            left, top, right, bottom = crop_rect
            handle_size = 10
            
            # Check corners first (they take precedence)
            if abs(x - left) <= handle_size and abs(y - top) <= handle_size:
                drag_mode = 'resize_nw'
            elif abs(x - right) <= handle_size and abs(y - top) <= handle_size:
                drag_mode = 'resize_ne'
            elif abs(x - left) <= handle_size and abs(y - bottom) <= handle_size:
                drag_mode = 'resize_sw'
            elif abs(x - right) <= handle_size and abs(y - bottom) <= handle_size:
                drag_mode = 'resize_se'
            # Then check edges
            elif abs(x - left) <= handle_size and y > top and y < bottom:
                drag_mode = 'resize_w'
            elif abs(x - right) <= handle_size and y > top and y < bottom:
                drag_mode = 'resize_e'
            elif abs(y - top) <= handle_size and x > left and x < right:
                drag_mode = 'resize_n'
            elif abs(y - bottom) <= handle_size and x > left and x < right:
                drag_mode = 'resize_s'
            # Finally check if inside the rectangle
            elif x > left and x < right and y > top and y < bottom:
                drag_mode = 'move'
            else:
                # Click outside the crop area - start a new selection
                drag_mode = 'resize_se'
                crop_rect[0] = x
                crop_rect[1] = y
                crop_rect[2] = x
                crop_rect[3] = y
            
            drag_start_x = x
            drag_start_y = y
            dragging = True
            
            # Update cursor based on drag mode
            update_cursor(x, y)
        
        def on_mouse_move(event):
            # Convert canvas coordinates to image coordinates
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            
            if dragging:
                # Handle dragging based on mode
                left, top, right, bottom = crop_rect
                
                if drag_mode == 'move':
                    # Move the entire rectangle
                    dx = x - drag_start_x
                    dy = y - drag_start_y
                    
                    # Update crop rectangle
                    crop_rect[0] = left + dx
                    crop_rect[1] = top + dy
                    crop_rect[2] = right + dx
                    crop_rect[3] = bottom + dy
                    
                    # Constrain to image bounds if needed
                    if constrain_var.get() and display_image:
                        img_width, img_height = display_image.size
                        
                        # Adjust if outside bounds
                        if crop_rect[0] < 0:
                            crop_rect[2] -= crop_rect[0]
                            crop_rect[0] = 0
                        if crop_rect[1] < 0:
                            crop_rect[3] -= crop_rect[1]
                            crop_rect[1] = 0
                        if crop_rect[2] > img_width:
                            crop_rect[0] -= (crop_rect[2] - img_width)
                            crop_rect[2] = img_width
                        if crop_rect[3] > img_height:
                            crop_rect[1] -= (crop_rect[3] - img_height)
                            crop_rect[3] = img_height
                    
                    # Update drag start position
                    drag_start_x = x
                    drag_start_y = y
                    
                elif drag_mode.startswith('resize_'):
                    # Resize the rectangle
                    if 'n' in drag_mode:
                        crop_rect[1] = y
                    if 's' in drag_mode:
                        crop_rect[3] = y
                    if 'w' in drag_mode:
                        crop_rect[0] = x
                    if 'e' in drag_mode:
                        crop_rect[2] = x
                    
                    # Ensure width and height are positive
                    if crop_rect[0] > crop_rect[2]:
                        crop_rect[0], crop_rect[2] = crop_rect[2], crop_rect[0]
                        drag_mode = drag_mode.replace('w', 'X').replace('e', 'w').replace('X', 'e')
                    
                    if crop_rect[1] > crop_rect[3]:
                        crop_rect[1], crop_rect[3] = crop_rect[3], crop_rect[1]
                        drag_mode = drag_mode.replace('n', 'X').replace('s', 'n').replace('X', 's')
                    
                    # Constrain to image bounds if needed
                    if constrain_var.get() and display_image:
                        img_width, img_height = display_image.size
                        crop_rect[0] = max(0, min(crop_rect[0], img_width - 10))
                        crop_rect[1] = max(0, min(crop_rect[1], img_height - 10))
                        crop_rect[2] = max(crop_rect[0] + 10, min(crop_rect[2], img_width))
                        crop_rect[3] = max(crop_rect[1] + 10, min(crop_rect[3], img_height))
                
                # Update the display
                update_crop_display()
            else:
                # Just update the cursor based on position
                update_cursor(x, y)
        
        def on_mouse_up(event):
            nonlocal dragging
            dragging = False
            
            # Apply aspect ratio constraint if needed
            aspect_ratio_name = aspect_ratio_var.get()
            if aspect_ratio_name != 'free':
                # Get aspect ratio values
                if aspect_ratio_name == 'custom':
                    width_ratio = custom_width_var.get()
                    height_ratio = custom_height_var.get()
                elif aspect_ratio_name == 'original':
                    width_ratio, height_ratio = image.size
                else:
                    width_ratio, height_ratio = self.aspect_ratio_presets.get(aspect_ratio_name, (0, 0))
                
                # Adjust crop rectangle to match aspect ratio
                if width_ratio > 0 and height_ratio > 0 and display_image:
                    display_width, display_height = display_image.size
                    crop_rect[:] = self.adjust_crop_for_aspect_ratio(
                        display_width, display_height, crop_rect, (width_ratio, height_ratio)
                    )
                    update_crop_display()
        
        def update_cursor(x, y):
            """Update the cursor based on position relative to crop rectangle."""
            if not dragging:
                # Check position relative to crop handles
                left, top, right, bottom = crop_rect
                handle_size = 10
                
                if abs(x - left) <= handle_size and abs(y - top) <= handle_size:
                    canvas.config(cursor="nw_resize")
                elif abs(x - right) <= handle_size and abs(y - top) <= handle_size:
                    canvas.config(cursor="ne_resize")
                elif abs(x - left) <= handle_size and abs(y - bottom) <= handle_size:
                    canvas.config(cursor="sw_resize")
                elif abs(x - right) <= handle_size and abs(y - bottom) <= handle_size:
                    canvas.config(cursor="se_resize")
                elif abs(x - left) <= handle_size and y > top and y < bottom:
                    canvas.config(cursor="w_resize")
                elif abs(x - right) <= handle_size and y > top and y < bottom:
                    canvas.config(cursor="e_resize")
                elif abs(y - top) <= handle_size and x > left and x < right:
                    canvas.config(cursor="n_resize")
                elif abs(y - bottom) <= handle_size and x > left and x < right:
                    canvas.config(cursor="s_resize")
                elif x > left and x < right and y > top and y < bottom:
                    canvas.config(cursor="fleur")  # Move cursor
                else:
                    canvas.config(cursor="")  # Default cursor
            else:
                # Keep the cursor appropriate for the current drag mode
                if drag_mode == 'move':
                    canvas.config(cursor="fleur")
                elif drag_mode == 'resize_nw':
                    canvas.config(cursor="nw_resize")
                elif drag_mode == 'resize_ne':
                    canvas.config(cursor="ne_resize")
                elif drag_mode == 'resize_sw':
                    canvas.config(cursor="sw_resize")
                elif drag_mode == 'resize_se':
                    canvas.config(cursor="se_resize")
                elif drag_mode == 'resize_n':
                    canvas.config(cursor="n_resize")
                elif drag_mode == 'resize_s':
                    canvas.config(cursor="s_resize")
                elif drag_mode == 'resize_w':
                    canvas.config(cursor="w_resize")
                elif drag_mode == 'resize_e':
                    canvas.config(cursor="e_resize")
        
        # Bind mouse events
        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)
        canvas.bind("<Motion>", on_mouse_move)  # For cursor updates
        
        # Function to handle aspect ratio changes
        def on_aspect_ratio_change(*args):
            if display_image:
                # Get the center of the current crop
                left, top, right, bottom = crop_rect
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                
                # Get the current dimensions
                width = right - left
                height = bottom - top
                
                # Get the new aspect ratio
                aspect_ratio_name = aspect_ratio_var.get()
                if aspect_ratio_name == 'free':
                    # No constraint, keep current dimensions
                    return
                
                # Get aspect ratio values
                if aspect_ratio_name == 'custom':
                    width_ratio = custom_width_var.get()
                    height_ratio = custom_height_var.get()
                elif aspect_ratio_name == 'original':
                    width_ratio, height_ratio = image.size
                else:
                    width_ratio, height_ratio = self.aspect_ratio_presets.get(aspect_ratio_name, (0, 0))
                
                # Calculate new dimensions
                if width_ratio > 0 and height_ratio > 0:
                    target_ratio = width_ratio / height_ratio
                    current_ratio = width / height
                    
                    if abs(current_ratio - target_ratio) > 0.01:
                        # Adjust dimensions to match the target ratio
                        if current_ratio > target_ratio:
                            # Too wide, adjust width
                            new_width = height * target_ratio
                            new_height = height
                        else:
                            # Too tall, adjust height
                            new_width = width
                            new_height = width / target_ratio
                        
                        # Calculate new crop rectangle
                        crop_rect[0] = center_x - new_width / 2
                        crop_rect[1] = center_y - new_height / 2
                        crop_rect[2] = center_x + new_width / 2
                        crop_rect[3] = center_y + new_height / 2
                        
                        # Constrain to image bounds if needed
                        if constrain_var.get() and display_image:
                            img_width, img_height = display_image.size
                            
                            # Adjust if outside bounds
                            if crop_rect[0] < 0:
                                crop_rect[2] -= crop_rect[0]
                                crop_rect[0] = 0
                            if crop_rect[1] < 0:
                                crop_rect[3] -= crop_rect[1]
                                crop_rect[1] = 0
                            if crop_rect[2] > img_width:
                                crop_rect[0] -= (crop_rect[2] - img_width)
                                crop_rect[2] = img_width
                            if crop_rect[3] > img_height:
                                crop_rect[1] -= (crop_rect[3] - img_height)
                                crop_rect[3] = img_height
                        
                        update_crop_display()
        
        # Bind aspect ratio change
        aspect_ratio_var.trace_add("write", on_aspect_ratio_change)
        
        # Function to handle rotation changes
        def on_rotation_change(*args):
            # Just update the display for preview
            # Actual rotation will be applied when the user clicks Apply
            update_crop_display()
        
        # Bind rotation change
        rotation_var.trace_add("write", on_rotation_change)
        
        # Function to handle auto-straighten
        def on_auto_straighten(*args):
            if auto_straighten_var.get() and image:
                # Apply auto-straighten
                straightened, angle = self.auto_straighten(image)
                
                if angle != 0:
                    # Update rotation value
                    rotation_var.set(angle)
                    
                    # Update the display
                    nonlocal display_image
                    display_image = straightened.copy()
                    display_image = display_image.resize(
                        (int(straightened.width * scale_factor), 
                         int(straightened.height * scale_factor)), 
                        Image.LANCZOS
                    )
                    
                    # Reset crop rectangle to full image
                    crop_rect[0] = 0
                    crop_rect[1] = 0
                    crop_rect[2] = display_image.width
                    crop_rect[3] = display_image.height
                    
                    update_crop_display()
        
        # Bind auto-straighten change
        auto_straighten_var.trace_add("write", on_auto_straighten)
        
        # Function to handle grid overlay changes
        def on_grid_change(*args):
            update_crop_display()
        
        # Bind grid change
        grid_var.trace_add("write", on_grid_change)
        
        # Function to handle flip changes
        def on_flip_change(*args):
            update_crop_display()
        
        # Bind flip changes
        flip_h_var.trace_add("write", on_flip_change)
        flip_v_var.trace_add("write", on_flip_change)
        
        # Initialize the display when the canvas is ready
        def on_canvas_configure(event):
            if not image_id:
                display_image_with_crop()
        
        canvas.bind("<Configure>", on_canvas_configure)
        
        # If the canvas is already visible, initialize now
        if canvas.winfo_width() > 1:
            display_image_with_crop()
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result[0]
    
    def create_cropper_panel(self, parent):
        """
        Create a panel with cropping controls for embedding in the main application.
        
        Args:
            parent: Parent widget
            
        Returns:
            Frame containing cropping controls
        """
        panel = ttk.LabelFrame(parent, text="Crop Image")
        
        # Aspect ratio selection
        aspect_frame = ttk.Frame(panel)
        aspect_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(aspect_frame, text="Aspect Ratio:").pack(side=tk.LEFT)
        
        self.aspect_ratio_var = StringVar(value=self.settings['aspect_ratio'])
        aspect_combo = ttk.Combobox(aspect_frame, textvariable=self.aspect_ratio_var, width=10)
        aspect_combo['values'] = ['free', 'original', '1:1', '16:9', '4:3', '3:2']
        aspect_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Rotation control
        rotation_frame = ttk.Frame(panel)
        rotation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(rotation_frame, text="Rotation:").pack(side=tk.LEFT)
        
        self.rotation_var = DoubleVar(value=self.settings['rotation'])
        rotation_scale = ttk.Scale(rotation_frame, from_=-45, to=45, variable=self.rotation_var, 
                                  orient=tk.HORIZONTAL)
        rotation_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        rotation_label = ttk.Label(rotation_frame, width=3)
        rotation_label.pack(side=tk.LEFT)
        
        # Update rotation label
        def update_rotation_label(*args):
            rotation_label.config(text=f"{int(self.rotation_var.get())}°")
        
        self.rotation_var.trace_add("write", update_rotation_label)
        update_rotation_label()  # Initialize
        
        # Flip controls
        flip_frame = ttk.Frame(panel)
        flip_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.flip_h_var = BooleanVar(value=self.settings['flip_horizontal'])
        ttk.Checkbutton(flip_frame, text="Flip Horizontal", 
                       variable=self.flip_h_var).pack(side=tk.LEFT, padx=(0, 10))
        
        self.flip_v_var = BooleanVar(value=self.settings['flip_vertical'])
        ttk.Checkbutton(flip_frame, text="Flip Vertical", 
                       variable=self.flip_v_var).pack(side=tk.LEFT)
        
        # Crop button
        ttk.Button(panel, text="Crop Image", command=self._on_crop_button_click).pack(
            fill=tk.X, padx=5, pady=5)
        
        return panel
    
    def _on_crop_button_click(self):
        """Handle crop button click in the panel."""
        if hasattr(self.controller, 'show_crop_dialog'):
            self.controller.show_crop_dialog()
    
    def get_panel_settings(self):
        """
        Get the current settings from the panel controls.
        
        Returns:
            Dictionary of current settings
        """
        settings = {}
        
        # Check if panel variables exist
        if hasattr(self, 'aspect_ratio_var'):
            settings['aspect_ratio'] = self.aspect_ratio_var.get()
        
        if hasattr(self, 'rotation_var'):
            settings['rotation'] = self.rotation_var.get()
        
        if hasattr(self, 'flip_h_var'):
            settings['flip_horizontal'] = self.flip_h_var.get()
        
        if hasattr(self, 'flip_v_var'):
            settings['flip_vertical'] = self.flip_v_var.get()
        
        return settings


def test_image_cropper():
    """Test function for the image cropper."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create a simple UI for testing
        root = tk.Tk()
        root.title("Image Cropper Test")
        root.geometry("1000x700")
        
        # Create the image cropper
        cropper = ImageCropper()
        
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
        
        # Create a frame for controls
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Button(control_frame, text="Open Image", command=open_image).pack(pady=10)
        
        # Show crop dialog
        def show_crop_dialog():
            if current_image[0] is None:
                return
                
            cropped = cropper.show_crop_dialog(root, current_image[0])
            if cropped:
                update_preview(cropped)
                current_image[0] = cropped
        
        ttk.Button(control_frame, text="Crop Image", command=show_crop_dialog).pack(pady=10)
        
        # Auto-straighten function
        def auto_straighten():
            if current_image[0] is None:
                return
                
            straightened, angle = cropper.auto_straighten(current_image[0])
            if straightened:
                update_preview(straightened)
                current_image[0] = straightened
                print(f"Auto-straightened by {angle:.2f} degrees")
        
        ttk.Button(control_frame, text="Auto-Straighten", command=auto_straighten).pack(pady=10)
        
        # Add the panel version of the cropper
        panel = cropper.create_cropper_panel(control_frame)
        panel.pack(fill=tk.X, pady=10)
        
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
        
        # Connect the panel to the controller
        cropper.controller = type('obj', (object,), {
            'show_crop_dialog': show_crop_dialog
        })
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_image_cropper()