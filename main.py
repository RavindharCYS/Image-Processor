import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
from tkinter.messagebox import showinfo, showerror
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
from rembg import remove
import io
from PIL.ImageDraw import Draw
from PIL.ImageColor import getrgb

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor Tool")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Variables
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.file_path = None
        self.bg_removed = False
        
        # Enhancement values
        self.brightness_val = tk.DoubleVar(value=1.0)
        self.contrast_val = tk.DoubleVar(value=1.0)
        self.sharpness_val = tk.DoubleVar(value=1.0)
        self.saturation_val = tk.DoubleVar(value=1.0)
        
        # Background colors
        self.bg_color = "#ffffff"
        self.gradient_color1 = "#ffffff"
        self.gradient_color2 = "#000000"
        
        # Create UI
        self.create_menu()
        self.create_main_layout()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Reset Image", command=self.reset_image)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_layout(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for tools
        left_panel = ttk.LabelFrame(main_frame, text="Tools")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Image display area
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.image_frame, bg="#f0f0f0")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars for canvas
        h_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Image placeholder
        self.image_label = ttk.Label(self.canvas, text="No image loaded")
        self.canvas.create_window((0, 0), window=self.image_label, anchor=tk.NW)
        
        # Tool buttons in left panel
        ttk.Button(left_panel, text="Open Image", command=self.open_image).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Remove Background", command=self.remove_background).pack(fill=tk.X, pady=5)
        
        # Background options
        bg_frame = ttk.LabelFrame(left_panel, text="Background Options")
        bg_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(bg_frame, text="Add Solid Background", command=self.add_solid_background).pack(fill=tk.X, pady=2)
        ttk.Button(bg_frame, text="Choose Color", command=self.choose_bg_color).pack(fill=tk.X, pady=2)
        
        ttk.Button(bg_frame, text="Add Gradient Background", command=self.add_gradient_background).pack(fill=tk.X, pady=2)
        ttk.Button(bg_frame, text="Choose Gradient Colors", command=self.choose_gradient_colors).pack(fill=tk.X, pady=2)
        
        # Gradient direction
        self.gradient_direction = tk.StringVar(value="horizontal")
        direction_frame = ttk.Frame(bg_frame)
        direction_frame.pack(fill=tk.X, pady=2)
        
        ttk.Radiobutton(direction_frame, text="Horizontal", variable=self.gradient_direction, 
                        value="horizontal").pack(side=tk.LEFT)
        ttk.Radiobutton(direction_frame, text="Vertical", variable=self.gradient_direction, 
                        value="vertical").pack(side=tk.LEFT)
        ttk.Radiobutton(direction_frame, text="Diagonal", variable=self.gradient_direction, 
                        value="diagonal").pack(side=tk.LEFT)
        
        # Enhancement options
        enhance_frame = ttk.LabelFrame(left_panel, text="Image Enhancement")
        enhance_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Brightness
        ttk.Label(enhance_frame, text="Brightness:").pack(anchor=tk.W)
        brightness_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                                    variable=self.brightness_val, command=lambda x: self.update_enhancement())
        brightness_scale.pack(fill=tk.X)
        
        # Contrast
        ttk.Label(enhance_frame, text="Contrast:").pack(anchor=tk.W)
        contrast_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                                  variable=self.contrast_val, command=lambda x: self.update_enhancement())
        contrast_scale.pack(fill=tk.X)
        
        # Sharpness
        ttk.Label(enhance_frame, text="Sharpness:").pack(anchor=tk.W)
        sharpness_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                                   variable=self.sharpness_val, command=lambda x: self.update_enhancement())
        sharpness_scale.pack(fill=tk.X)
        
        # Saturation
        ttk.Label(enhance_frame, text="Saturation:").pack(anchor=tk.W)
        saturation_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                                    variable=self.saturation_val, command=lambda x: self.update_enhancement())
        saturation_scale.pack(fill=tk.X)
        
        # Crop options
        crop_frame = ttk.LabelFrame(left_panel, text="Crop Image")
        crop_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(crop_frame, text="Start Crop Mode", command=self.start_crop_mode).pack(fill=tk.X, pady=2)
        ttk.Button(crop_frame, text="Apply Crop", command=self.apply_crop).pack(fill=tk.X, pady=2)
        
        # Save options
        save_frame = ttk.LabelFrame(left_panel, text="Save Options")
        save_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.save_format = tk.StringVar(value="png")
        ttk.Radiobutton(save_frame, text="PNG (with transparency)", variable=self.save_format, 
                       value="png").pack(anchor=tk.W)
        ttk.Radiobutton(save_frame, text="JPG", variable=self.save_format, 
                       value="jpg").pack(anchor=tk.W)
        
        ttk.Button(save_frame, text="Save Image", command=self.save_image).pack(fill=tk.X, pady=5)
        
        # Reset button
        ttk.Button(left_panel, text="Reset Image", command=self.reset_image).pack(fill=tk.X, pady=10)
        
        # Crop variables
        self.crop_mode = False
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_rect = None
        
        # Bind canvas events for cropping
        self.canvas.bind("<ButtonPress-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_release)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                self.file_path = file_path
                self.original_image = Image.open(file_path)
                self.processed_image = self.original_image.copy()
                self.current_image = self.original_image.copy()
                self.bg_removed = False
                
                # Reset enhancement values
                self.brightness_val.set(1.0)
                self.contrast_val.set(1.0)
                self.sharpness_val.set(1.0)
                self.saturation_val.set(1.0)
                
                self.display_image()
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                showerror("Error", f"Could not open image: {str(e)}")
    
    def display_image(self):
        if self.current_image:
            # Resize for display if needed
            display_image = self.current_image.copy()
            
            # Calculate new size to fit canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
                img_width, img_height = display_image.size
                
                # Calculate scale factor
                width_ratio = canvas_width / img_width
                height_ratio = canvas_height / img_height
                scale_factor = min(width_ratio, height_ratio)
                
                if scale_factor < 1:  # Only scale down, not up
                    new_width = int(img_width * scale_factor)
                    new_height = int(img_height * scale_factor)
                    display_image = display_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.tk_image = ImageTk.PhotoImage(display_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Configure scrollregion
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def remove_background(self):
        if self.current_image:
            try:
                self.status_var.set("Removing background... This may take a moment.")
                self.root.update()
                
                # Use rembg to remove background
                img_data = io.BytesIO()
                self.processed_image.save(img_data, format='PNG')
                img_data.seek(0)
                
                output = remove(img_data.read())
                self.processed_image = Image.open(io.BytesIO(output)).convert("RGBA")
                self.current_image = self.processed_image.copy()
                self.bg_removed = True
                
                self.display_image()
                self.status_var.set("Background removed successfully")
            except Exception as e:
                showerror("Error", f"Failed to remove background: {str(e)}")
                self.status_var.set("Error removing background")
    
    def choose_bg_color(self):
        color = colorchooser.askcolor(title="Choose background color", initialcolor=self.bg_color)
        if color[1]:
            self.bg_color = color[1]
    
    def choose_gradient_colors(self):
        color1 = colorchooser.askcolor(title="Choose first gradient color", initialcolor=self.gradient_color1)
        if color1[1]:
            self.gradient_color1 = color1[1]
            
        color2 = colorchooser.askcolor(title="Choose second gradient color", initialcolor=self.gradient_color2)
        if color2[1]:
            self.gradient_color2 = color2[1]
    
    def add_solid_background(self):
        if not self.current_image or not self.bg_removed:
            showinfo("Info", "Please remove the background first")
            return
        
        try:
            # Create a solid color background
            bg = Image.new('RGBA', self.processed_image.size, self.bg_color)
            
            # Composite the image with the background
            result = Image.alpha_composite(bg, self.processed_image)
            self.current_image = result
            
            self.display_image()
            self.status_var.set(f"Added solid background: {self.bg_color}")
        except Exception as e:
            showerror("Error", f"Failed to add background: {str(e)}")
    
    def add_gradient_background(self):
        if not self.current_image or not self.bg_removed:
            showinfo("Info", "Please remove the background first")
            return
        
        try:
            width, height = self.processed_image.size
            gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = Draw(gradient)
            
            direction = self.gradient_direction.get()
            color1 = getrgb(self.gradient_color1)
            color2 = getrgb(self.gradient_color2)
            
            # Add alpha channel
            color1 = color1 + (255,)
            color2 = color2 + (255,)
            
            if direction == "horizontal":
                for x in range(width):
                    # Calculate gradient color at this position
                    r = int(color1[0] + (color2[0] - color1[0]) * x / width)
                    g = int(color1[1] + (color2[1] - color1[1]) * x / width)
                    b = int(color1[2] + (color2[2] - color1[2]) * x / width)
                    
                    draw.line([(x, 0), (x, height)], fill=(r, g, b, 255))
            
            elif direction == "vertical":
                for y in range(height):
                    # Calculate gradient color at this position
                    r = int(color1[0] + (color2[0] - color1[0]) * y / height)
                    g = int(color1[1] + (color2[1] - color1[1]) * y / height)
                    b = int(color1[2] + (color2[2] - color1[2]) * y / height)
                    
                    draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
            
            elif direction == "diagonal":
                for i in range(width + height):
                    # Calculate gradient color at this position
                    r = int(color1[0] + (color2[0] - color1[0]) * i / (width + height))
                    g = int(color1[1] + (color2[1] - color1[1]) * i / (width + height))
                    b = int(color1[2] + (color2[2] - color1[2]) * i / (width + height))
                    
                    draw.line([(0, i), (i, 0)], fill=(r, g, b, 255))
            
            # Composite the gradient with the image
            result = Image.alpha_composite(gradient, self.processed_image)
            self.current_image = result
            
            self.display_image()
            self.status_var.set(f"Added {direction} gradient background")
        except Exception as e:
            showerror("Error", f"Failed to add gradient background: {str(e)}")
    
    def update_enhancement(self):
        if self.processed_image:
            try:
                # Start with the processed image (after bg removal if applicable)
                enhanced = self.processed_image.copy()
                
                # Apply enhancements
                if enhanced.mode != 'RGBA':
                    enhanced = enhanced.convert('RGBA')
                
                # Split the image into bands
                r, g, b, a = enhanced.split()
                rgb_image = Image.merge('RGB', (r, g, b))
                
                # Apply enhancements to RGB channels
                rgb_image = ImageEnhance.Brightness(rgb_image).enhance(self.brightness_val.get())
                rgb_image = ImageEnhance.Contrast(rgb_image).enhance(self.contrast_val.get())
                rgb_image = ImageEnhance.Sharpness(rgb_image).enhance(self.sharpness_val.get())
                rgb_image = ImageEnhance.Color(rgb_image).enhance(self.saturation_val.get())
                
                # Recombine with alpha channel
                r, g, b = rgb_image.split()
                enhanced = Image.merge('RGBA', (r, g, b, a))
                
                self.current_image = enhanced
                self.display_image()
                self.status_var.set("Image enhanced")
            except Exception as e:
                showerror("Error", f"Failed to enhance image: {str(e)}")
    
    def start_crop_mode(self):
        if self.current_image:
            self.crop_mode = True
            self.status_var.set("Crop mode: Click and drag to select crop area")
    
    def on_crop_start(self, event):
        if self.crop_mode and self.current_image:
            # Get canvas coordinates
            self.crop_start_x = self.canvas.canvasx(event.x)
            self.crop_start_y = self.canvas.canvasy(event.y)
            
            # Create rectangle
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            
            self.crop_rect = self.canvas.create_rectangle(
                self.crop_start_x, self.crop_start_y, 
                self.crop_start_x, self.crop_start_y,
                outline="red", width=2
            )
    
    def on_crop_motion(self, event):
        if self.crop_mode and self.crop_rect:
            # Update rectangle size
            current_x = self.canvas.canvasx(event.x)
            current_y = self.canvas.canvasy(event.y)
            
            self.canvas.coords(
                self.crop_rect,
                self.crop_start_x, self.crop_start_y,
                current_x, current_y
            )
    
    def on_crop_release(self, event):
        # Crop selection is complete
        pass

    def apply_crop(self):
        if self.crop_mode and self.crop_rect and self.current_image:
            try:
                # Get the coordinates of the crop rectangle
                x1, y1, x2, y2 = self.canvas.coords(self.crop_rect)
                
                # Convert canvas coordinates to image coordinates
                # This accounts for any scaling that might have been applied
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                img_width, img_height = self.current_image.size
                
                # Calculate the scale factor
                width_ratio = img_width / canvas_width
                height_ratio = img_height / canvas_height
                
                # Use the larger ratio to ensure we stay within image bounds
                scale_factor = max(width_ratio, height_ratio)
                
                # Convert to image coordinates
                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)
                
                # Ensure coordinates are in the correct order
                left = min(x1, x2)
                top = min(y1, y2)
                right = max(x1, x2)
                bottom = max(y1, y2)
                
                # Ensure coordinates are within image bounds
                left = max(0, left)
                top = max(0, top)
                right = min(img_width, right)
                bottom = min(img_height, bottom)
                
                # Crop the image
                cropped_image = self.current_image.crop((left, top, right, bottom))
                
                # Update the current image
                self.current_image = cropped_image
                self.processed_image = cropped_image
                
                # Exit crop mode
                self.crop_mode = False
                if self.crop_rect:
                    self.canvas.delete(self.crop_rect)
                    self.crop_rect = None
                
                # Display the cropped image
                self.display_image()
                self.status_var.set("Image cropped successfully")
            except Exception as e:
                showerror("Error", f"Failed to crop image: {str(e)}")
                self.status_var.set("Error cropping image")
    
    def save_image(self):
        if self.current_image:
            file_format = self.save_format.get()
            default_ext = f".{file_format}"
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=default_ext,
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    # If saving as JPG, convert to RGB (remove alpha channel)
                    if file_format.lower() == 'jpg':
                        if self.current_image.mode == 'RGBA':
                            # Create a white background
                            bg = Image.new('RGB', self.current_image.size, (255, 255, 255))
                            bg.paste(self.current_image, mask=self.current_image.split()[3])
                            bg.save(file_path, 'JPEG', quality=95)
                        else:
                            self.current_image.convert('RGB').save(file_path, 'JPEG', quality=95)
                    else:
                        self.current_image.save(file_path)
                    
                    self.status_var.set(f"Image saved as {os.path.basename(file_path)}")
                    showinfo("Success", "Image saved successfully!")
                except Exception as e:
                    showerror("Error", f"Failed to save image: {str(e)}")
                    self.status_var.set("Error saving image")
    
    def reset_image(self):
        if self.original_image:
            self.processed_image = self.original_image.copy()
            self.current_image = self.original_image.copy()
            self.bg_removed = False
            
            # Reset enhancement values
            self.brightness_val.set(1.0)
            self.contrast_val.set(1.0)
            self.sharpness_val.set(1.0)
            self.saturation_val.set(1.0)
            
            # Exit crop mode if active
            self.crop_mode = False
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
                self.crop_rect = None
            
            self.display_image()
            self.status_var.set("Image reset to original")
    
    def show_about(self):
        about_text = """
        Image Processor Tool
        
        Features:
        - Background removal
        - Add solid or gradient backgrounds
        - Image enhancement
        - Crop functionality
        - Save as PNG or JPG
        
        Created with Python and Tkinter
        """
        showinfo("About", about_text)


def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    
    # Update the display when window is resized
    def on_resize(event):
        if hasattr(app, 'current_image') and app.current_image:
            app.display_image()
    
    root.bind("<Configure>", on_resize)
    
    root.mainloop()


if __name__ == "__main__":
    main()