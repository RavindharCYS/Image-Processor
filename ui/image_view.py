import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math

class ImageView:
    """
    Advanced image viewing component with zooming, panning, and interactive features.
    """
    
    def __init__(self, parent, controller):
        """
        Initialize the image view component.
        
        Args:
            parent: The parent widget
            controller: The controller object that handles the application logic
        """
        self.parent = parent
        self.controller = controller
        
        # Create the main frame
        self.frame = ttk.LabelFrame(parent, text="Image Preview")
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the canvas for image display
        self.canvas = tk.Canvas(self.frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        # Image display variables
        self.image_id = None
        self.tk_image = None
        self.current_image = None
        self.original_image = None
        self.zoom_level = 1.0
        self.max_zoom = 5.0
        self.min_zoom = 0.1
        
        # Interaction state variables
        self.panning = False
        self.selecting = False
        self.selection_rect = None
        self.start_x = 0
        self.start_y = 0
        self.crop_mode = False
        
        # Information display
        self.info_text_id = None
        
        # Create placeholder text
        self.placeholder_id = self.canvas.create_text(
            10, 10, anchor=tk.NW, text="No image loaded", fill="gray", font=("Arial", 12)
        )
        
        # Create toolbar at the bottom of the image view
        self.create_view_toolbar()
        
        # Set up event bindings
        self.setup_bindings()
        
        # Make the controller aware of this view
        if hasattr(controller, 'set_image_view'):
            controller.set_image_view(self)
        else:
            # Store canvas reference in controller for backward compatibility
            controller.canvas = self.canvas
    
    def create_view_toolbar(self):
        """Create a toolbar at the bottom of the image view with zoom controls."""
        toolbar_frame = ttk.Frame(self.frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, before=self.h_scrollbar)
        
        # Zoom out button
        self.zoom_out_btn = ttk.Button(toolbar_frame, text="−", width=2, command=self.zoom_out)
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Zoom level indicator
        self.zoom_var = tk.StringVar(value="100%")
        zoom_label = ttk.Label(toolbar_frame, textvariable=self.zoom_var, width=8)
        zoom_label.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Zoom in button
        self.zoom_in_btn = ttk.Button(toolbar_frame, text="+", width=2, command=self.zoom_in)
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Reset zoom button
        self.reset_zoom_btn = ttk.Button(toolbar_frame, text="Fit", width=4, command=self.reset_zoom)
        self.reset_zoom_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Actual size button
        self.actual_size_btn = ttk.Button(toolbar_frame, text="1:1", width=4, command=self.actual_size)
        self.actual_size_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Image information
        self.info_var = tk.StringVar(value="")
        info_label = ttk.Label(toolbar_frame, textvariable=self.info_var)
        info_label.pack(side=tk.RIGHT, padx=5, pady=2)
    
    def setup_bindings(self):
        """Set up event bindings for the canvas."""
        # Mouse wheel for zooming
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)    # Linux scroll down
        
        # Middle button (or wheel click) for panning
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)
        
        # Alt + Left button can also be used for panning (alternative)
        self.canvas.bind("<Alt-ButtonPress-1>", self.start_pan)
        self.canvas.bind("<Alt-B1-Motion>", self.pan_image)
        self.canvas.bind("<Alt-ButtonRelease-1>", self.stop_pan)
        
        # Left button for selection/cropping
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Right-click context menu
        self.canvas.bind("<ButtonPress-3>", self.show_context_menu)
        
        # Key bindings
        self.canvas.bind("<KeyPress>", self.on_key_press)
        self.canvas.focus_set()  # Ensure canvas can receive key events
    
    def set_image(self, image):
        """
        Set the image to display.
        
        Args:
            image: PIL Image object to display
        """
        if image:
            self.original_image = image
            self.current_image = image.copy()
            self.display_image()
            self.update_info()
            self.canvas.delete(self.placeholder_id)
        else:
            self.clear_image()
    
    def clear_image(self):
        """Clear the current image."""
        if self.image_id:
            self.canvas.delete(self.image_id)
            self.image_id = None
            self.tk_image = None
            self.current_image = None
            self.original_image = None
            
            # Reset zoom
            self.zoom_level = 1.0
            self.zoom_var.set("100%")
            
            # Show placeholder
            if not self.placeholder_id:
                self.placeholder_id = self.canvas.create_text(
                    10, 10, anchor=tk.NW, text="No image loaded", fill="gray", font=("Arial", 12)
                )
            else:
                self.canvas.itemconfigure(self.placeholder_id, state=tk.NORMAL)
            
            # Clear info
            self.info_var.set("")
    
    def display_image(self):
        """Display the current image with the current zoom level."""
        if not self.current_image:
            return
        
        # Calculate the new size based on zoom level
        original_width, original_height = self.current_image.size
        new_width = int(original_width * self.zoom_level)
        new_height = int(original_height * self.zoom_level)
        
        # Resize the image
        if self.zoom_level != 1.0:
            resized_image = self.current_image.resize((new_width, new_height), Image.LANCZOS)
        else:
            resized_image = self.current_image
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        # Update or create image on canvas
        if self.image_id:
            self.canvas.itemconfigure(self.image_id, image=self.tk_image)
        else:
            self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Update canvas scroll region
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        # Update zoom indicator
        self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
        
        # Hide placeholder if visible
        if self.placeholder_id:
            self.canvas.itemconfigure(self.placeholder_id, state=tk.HIDDEN)
    
    def update_info(self):
        """Update the image information display."""
        if self.current_image:
            width, height = self.current_image.size
            mode = self.current_image.mode
            self.info_var.set(f"{width}×{height} pixels | {mode}")
        else:
            self.info_var.set("")
    
    def zoom_in(self, event=None):
        """Zoom in on the image."""
        if self.current_image:
            self.zoom_level = min(self.zoom_level * 1.25, self.max_zoom)
            self.display_image()
    
    def zoom_out(self, event=None):
        """Zoom out from the image."""
        if self.current_image:
            self.zoom_level = max(self.zoom_level / 1.25, self.min_zoom)
            self.display_image()
    
    def reset_zoom(self, event=None):
        """Reset zoom to fit the image in the canvas."""
        if not self.current_image:
            return
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet properly initialized
            return
        
        # Get image size
        img_width, img_height = self.current_image.size
        
        # Calculate zoom level to fit
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        
        # Use the smaller ratio to ensure the entire image fits
        self.zoom_level = min(width_ratio, height_ratio) * 0.95  # 5% margin
        
        self.display_image()
        
        # Center the image
        self.center_image()
    
    def actual_size(self, event=None):
        """Set zoom level to 100% (actual size)."""
        if self.current_image:
            self.zoom_level = 1.0
            self.display_image()
    
    def center_image(self):
        """Center the image in the canvas."""
        if not self.current_image or not self.image_id:
            return
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get image size with current zoom
        img_width = int(self.current_image.width * self.zoom_level)
        img_height = int(self.current_image.height * self.zoom_level)
        
        # Calculate center position
        x = max(0, (canvas_width - img_width) / 2)
        y = max(0, (canvas_height - img_height) / 2)
        
        # Move image
        self.canvas.coords(self.image_id, x, y)
        
        # Update scroll region
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def on_mousewheel(self, event):
        """Handle mouse wheel events for zooming."""
        if not self.current_image:
            return
        
        # Determine zoom direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            # Zoom in
            zoom_factor = 1.1
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            # Zoom out
            zoom_factor = 0.9
        else:
            return
        
        # Calculate new zoom level
        new_zoom = self.zoom_level * zoom_factor
        
        # Enforce zoom limits
        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return
        
        # Get mouse position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Get current scroll position
        x_scroll = self.canvas.canvasx(0)
        y_scroll = self.canvas.canvasy(0)
        
        # Calculate the position relative to the image
        rel_x = x - x_scroll
        rel_y = y - y_scroll
        
        # Update zoom level
        self.zoom_level = new_zoom
        
        # Redisplay image
        self.display_image()
        
        # Calculate new scroll position to keep the point under cursor
        new_x_scroll = x - rel_x * zoom_factor
        new_y_scroll = y - rel_y * zoom_factor
        
        # Set new scroll position
        self.canvas.xview_moveto(new_x_scroll / self.canvas.bbox(tk.ALL)[2])
        self.canvas.yview_moveto(new_y_scroll / self.canvas.bbox(tk.ALL)[3])
    
    def start_pan(self, event):
        """Start panning the image."""
        self.canvas.config(cursor="fleur")  # Change cursor to indicate panning
        self.panning = True
        self.start_x = event.x
        self.start_y = event.y
    
    def pan_image(self, event):
        """Pan the image as the mouse moves."""
        if not self.panning:
            return
        
        # Calculate the distance moved
        dx = self.start_x - event.x
        dy = self.start_y - event.y
        
        # Scroll the canvas
        self.canvas.xview_scroll(dx, "units")
        self.canvas.yview_scroll(dy, "units")
        
        # Update start position
        self.start_x = event.x
        self.start_y = event.y
    
    def stop_pan(self, event):
        """Stop panning the image."""
        self.panning = False
        self.canvas.config(cursor="")  # Reset cursor
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if self.crop_mode and self.current_image:
            self.selecting = True
            self.start_x = self.canvas.canvasx(event.x)
            self.start_y = self.canvas.canvasy(event.y)
            
            # Create selection rectangle
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            
            self.selection_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline="red", width=2, dash=(4, 4)
            )
    
    def on_mouse_move(self, event):
        """Handle mouse movement."""
        if self.selecting and self.selection_rect:
            # Update selection rectangle
            current_x = self.canvas.canvasx(event.x)
            current_y = self.canvas.canvasy(event.y)
            
            self.canvas.coords(
                self.selection_rect,
                self.start_x, self.start_y,
                current_x, current_y
            )
            
            # Show dimensions in status bar if available
            if hasattr(self.controller, 'status_var'):
                width = abs(current_x - self.start_x) / self.zoom_level
                height = abs(current_y - self.start_y) / self.zoom_level
                self.controller.status_var.set(f"Selection: {int(width)}×{int(height)} pixels")
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if self.selecting:
            self.selecting = False
            
            # If we have a controller with a crop method, call it
            if self.crop_mode and hasattr(self.controller, 'apply_crop'):
                # The controller will handle the actual cropping
                pass
    
    def get_crop_coordinates(self):
        """
        Get the coordinates of the current selection rectangle.
        
        Returns:
            Tuple of (left, top, right, bottom) in image coordinates
        """
        if not self.selection_rect:
            return None
        
        # Get canvas coordinates
        x1, y1, x2, y2 = self.canvas.coords(self.selection_rect)
        
        # Convert to image coordinates (accounting for zoom)
        left = min(x1, x2) / self.zoom_level
        top = min(y1, y2) / self.zoom_level
        right = max(x1, x2) / self.zoom_level
        bottom = max(y1, y2) / self.zoom_level
        
        return (int(left), int(top), int(right), int(bottom))
    
    def clear_selection(self):
        """Clear the current selection rectangle."""
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
    
    def set_crop_mode(self, enabled):
        """
        Enable or disable crop mode.
        
        Args:
            enabled: Boolean indicating whether crop mode should be enabled
        """
        self.crop_mode = enabled
        
        if enabled:
            self.canvas.config(cursor="crosshair")
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set("Crop mode: Click and drag to select area")
        else:
            self.canvas.config(cursor="")
            self.clear_selection()
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set("Ready")
    
    def show_context_menu(self, event):
        """Show a context menu on right-click."""
        if not self.current_image:
            return
        
        # Create a context menu
        context_menu = tk.Menu(self.canvas, tearoff=0)
        
        # Add menu items
        context_menu.add_command(label="Zoom In", command=self.zoom_in)
        context_menu.add_command(label="Zoom Out", command=self.zoom_out)
        context_menu.add_command(label="Actual Size (100%)", command=self.actual_size)
        context_menu.add_command(label="Fit to Window", command=self.reset_zoom)
        
        context_menu.add_separator()
        
        # Crop option
        if self.crop_mode:
            context_menu.add_command(label="Cancel Crop", 
                                    command=lambda: self.set_crop_mode(False))
        else:
            context_menu.add_command(label="Crop Image", 
                                    command=lambda: self.set_crop_mode(True))
        
        context_menu.add_separator()
        
        # Copy to clipboard option
        context_menu.add_command(label="Copy Image", command=self.copy_to_clipboard)
        
        # Save option
        context_menu.add_command(label="Save Image", command=self.controller.save_image)
        
        # Display the menu
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            # Make sure to release the grab
            context_menu.grab_release()
        
    def copy_to_clipboard(self):
        """Copy the current image to clipboard."""
        if not self.current_image:
            return
            
        # This is platform-dependent and might require additional libraries
        try:
            import io
            import subprocess
            
            # For Windows
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set("Copying image to clipboard...")
            
            # Save image to a temporary file
            temp_file = io.BytesIO()
            self.current_image.save(temp_file, format='PNG')
            temp_file.seek(0)
            
            # Try to use platform-specific methods
            import platform
            system = platform.system().lower()
            
            if system == 'windows':
                import win32clipboard
                
                # Convert to DIB (Device Independent Bitmap)
                output = io.BytesIO()
                self.current_image.convert('RGB').save(output, 'BMP')
                data = output.getvalue()[14:]  # The file header is 14 bytes
                output.close()
                
                # Send to clipboard
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                
                if hasattr(self.controller, 'status_var'):
                    self.controller.status_var.set("Image copied to clipboard")
                
            elif system == 'darwin':  # macOS
                # Use pbcopy
                process = subprocess.Popen(['pbcopy', '-Prefer', 'png'], 
                                          stdin=subprocess.PIPE, close_fds=True)
                process.communicate(input=temp_file.getvalue())
                
                if hasattr(self.controller, 'status_var'):
                    self.controller.status_var.set("Image copied to clipboard")
                
            elif system == 'linux':
                # Try xclip or wl-copy for Wayland
                try:
                    # First try xclip (X11)
                    process = subprocess.Popen(['xclip', '-selection', 'clipboard', '-t', 'image/png'], 
                                              stdin=subprocess.PIPE, close_fds=True)
                    process.communicate(input=temp_file.getvalue())
                except FileNotFoundError:
                    try:
                        # Then try wl-copy (Wayland)
                        process = subprocess.Popen(['wl-copy', '-t', 'image/png'], 
                                                  stdin=subprocess.PIPE, close_fds=True)
                        process.communicate(input=temp_file.getvalue())
                    except FileNotFoundError:
                        if hasattr(self.controller, 'status_var'):
                            self.controller.status_var.set("Clipboard not supported. Install xclip or wl-copy.")
                        return
                
                if hasattr(self.controller, 'status_var'):
                    self.controller.status_var.set("Image copied to clipboard")
            else:
                if hasattr(self.controller, 'status_var'):
                    self.controller.status_var.set("Clipboard not supported on this platform")
        
        except ImportError as e:
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Clipboard error: Missing module - {str(e)}")
        except Exception as e:
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Failed to copy to clipboard: {str(e)}")
    
    def on_key_press(self, event):
        """Handle key press events."""
        if not self.current_image:
            return
            
        key = event.keysym.lower()
        
        # Zoom controls
        if key == 'plus' or key == 'equal':
            self.zoom_in()
        elif key == 'minus':
            self.zoom_out()
        elif key == '0':
            self.actual_size()
        elif key == 'f':
            self.reset_zoom()
            
        # Panning with arrow keys
        elif key == 'left':
            self.canvas.xview_scroll(-1, "units")
        elif key == 'right':
            self.canvas.xview_scroll(1, "units")
        elif key == 'up':
            self.canvas.yview_scroll(-1, "units")
        elif key == 'down':
            self.canvas.yview_scroll(1, "units")
            
        # Crop mode toggle
        elif key == 'c':
            self.set_crop_mode(not self.crop_mode)
            
        # Escape to cancel crop
        elif key == 'escape' and self.crop_mode:
            self.set_crop_mode(False)
            
        # Copy to clipboard
        elif key == 'control_l' and event.char == '\x03':  # Ctrl+C
            self.copy_to_clipboard()
    
    def show_image_info(self, event=None):
        """Show detailed information about the image."""
        if not self.current_image:
            return
            
        # Create a new toplevel window
        info_window = tk.Toplevel(self.parent)
        info_window.title("Image Information")
        info_window.geometry("400x300")
        info_window.resizable(False, False)
        
        # Make it modal
        info_window.transient(self.parent)
        info_window.grab_set()
        
        # Image properties
        width, height = self.current_image.size
        mode = self.current_image.mode
        format_str = getattr(self.current_image, 'format', 'Unknown')
        
        # Create a frame with padding
        frame = ttk.Frame(info_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add information
        ttk.Label(frame, text="Image Properties", font=("Arial", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        props = [
            ("Dimensions:", f"{width} × {height} pixels"),
            ("Color Mode:", mode),
            ("Format:", format_str),
            ("Zoom Level:", f"{int(self.zoom_level * 100)}%"),
            ("File Size:", self._get_file_size()),
            ("Transparency:", "Yes" if self.current_image.mode == 'RGBA' else "No"),
        ]
        
        for i, (label, value) in enumerate(props):
            ttk.Label(frame, text=label, font=("Arial", 10, "bold")).grid(
                row=i+1, column=0, sticky=tk.W, pady=2)
            ttk.Label(frame, text=value).grid(
                row=i+1, column=1, sticky=tk.W, pady=2)
        
        # Histogram information if available
        if hasattr(self.current_image, 'histogram'):
            ttk.Label(frame, text="Histogram:", font=("Arial", 10, "bold")).grid(
                row=len(props)+1, column=0, sticky=tk.W, pady=(10, 2))
            
            # Create a simple histogram visualization
            histogram_frame = ttk.Frame(frame)
            histogram_frame.grid(row=len(props)+1, column=1, sticky=tk.W, pady=(10, 2))
            
            # This would be a placeholder for a real histogram visualization
            ttk.Label(histogram_frame, text="Histogram visualization would go here").pack()
        
        # Close button
        ttk.Button(frame, text="Close", command=info_window.destroy).grid(
            row=len(props)+2, column=0, columnspan=2, pady=(10, 0))
        
        # Center the window on the parent
        info_window.update_idletasks()
        x = self.parent.winfo_rootx() + (self.parent.winfo_width() - info_window.winfo_width()) // 2
        y = self.parent.winfo_rooty() + (self.parent.winfo_height() - info_window.winfo_height()) // 2
        info_window.geometry(f"+{x}+{y}")
    
    def _get_file_size(self):
        """Get the file size of the current image."""
        if not self.current_image:
            return "Unknown"
            
        # Save to a BytesIO object to get size
        temp_file = io.BytesIO()
        self.current_image.save(temp_file, format='PNG')
        size_bytes = temp_file.tell()
        
        # Format size
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    def rotate_image(self, degrees):
        """
        Rotate the image by the specified degrees.
        
        Args:
            degrees: Rotation angle in degrees (clockwise)
        """
        if not self.current_image:
            return
            
        try:
            # Rotate the image
            rotated = self.current_image.rotate(-degrees, resample=Image.BICUBIC, expand=True)
            
            # Update the image
            self.current_image = rotated
            
            # If we have a controller with an update method, call it
            if hasattr(self.controller, 'set_processed_image'):
                self.controller.set_processed_image(rotated)
            
            # Display the rotated image
            self.display_image()
            
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Image rotated {degrees}°")
        except Exception as e:
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Failed to rotate image: {str(e)}")
    
    def flip_image(self, direction):
        """
        Flip the image horizontally or vertically.
        
        Args:
            direction: Either 'horizontal' or 'vertical'
        """
        if not self.current_image:
            return
            
        try:
            if direction == 'horizontal':
                flipped = self.current_image.transpose(Image.FLIP_LEFT_RIGHT)
                flip_type = "horizontally"
            elif direction == 'vertical':
                flipped = self.current_image.transpose(Image.FLIP_TOP_BOTTOM)
                flip_type = "vertically"
            else:
                return
            
            # Update the image
            self.current_image = flipped
            
            # If we have a controller with an update method, call it
            if hasattr(self.controller, 'set_processed_image'):
                self.controller.set_processed_image(flipped)
            
            # Display the flipped image
            self.display_image()
            
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Image flipped {flip_type}")
        except Exception as e:
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Failed to flip image: {str(e)}")
    
    def add_overlay_text(self, text, position='center', color='white', font_size=20, opacity=0.8):
        """
        Add text overlay to the image.
        
        Args:
            text: Text to add
            position: Position ('center', 'top', 'bottom', 'top-left', etc.)
            color: Text color
            font_size: Font size
            opacity: Text opacity (0-1)
        """
        if not self.current_image:
            return
            
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create a copy of the image
            img_with_text = self.current_image.copy()
            
            # Create a drawing context
            draw = ImageDraw.Draw(img_with_text)
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            
            # Calculate text size
            text_width, text_height = draw.textsize(text, font=font)
            
            # Calculate position
            width, height = img_with_text.size
            
            if position == 'center':
                x = (width - text_width) // 2
                y = (height - text_height) // 2
            elif position == 'top':
                x = (width - text_width) // 2
                y = 10
            elif position == 'bottom':
                x = (width - text_width) // 2
                y = height - text_height - 10
            elif position == 'top-left':
                x = 10
                y = 10
            elif position == 'top-right':
                x = width - text_width - 10
                y = 10
            elif position == 'bottom-left':
                x = 10
                y = height - text_height - 10
            elif position == 'bottom-right':
                x = width - text_width - 10
                y = height - text_height - 10
            else:
                x = (width - text_width) // 2
                y = (height - text_height) // 2
            
            # Add a semi-transparent background for the text
            padding = 5
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=(0, 0, 0, int(opacity * 128))
            )
            
            # Draw the text
            draw.text((x, y), text, fill=color, font=font)
            
            # Update the image
            self.current_image = img_with_text
            
            # If we have a controller with an update method, call it
            if hasattr(self.controller, 'set_processed_image'):
                self.controller.set_processed_image(img_with_text)
            
            # Display the image with text
            self.display_image()
            
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Added text overlay")
        except Exception as e:
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Failed to add text: {str(e)}")