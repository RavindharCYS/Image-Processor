import tkinter as tk
from tkinter import ttk, Frame, LabelFrame, Canvas, Label, Button, Radiobutton, Scale
from tkinter import StringVar, DoubleVar, HORIZONTAL, VERTICAL, SUNKEN, W, X, Y, BOTH, LEFT, RIGHT, BOTTOM, NW, ALL

class MainWindow:
    def __init__(self, root, controller):
        """
        Initialize the main window of the application.
        
        Args:
            root: The tkinter root window
            controller: The controller object that handles the application logic
        """
        self.root = root
        self.controller = controller
        
        # Configure the root window
        self.root.title("Image Processor Tool")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Set up the UI components
        self.create_menu()
        self.create_main_layout()
        self.create_status_bar()
        self.setup_bindings()
        
        # Apply styling
        self.apply_styling()
    
    def create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.controller.open_image)
        file_menu.add_command(label="Save", command=self.controller.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Reset Image", command=self.controller.reset_image)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.controller.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_layout(self):
        """Create the main layout of the application."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for tools
        self.create_tools_panel(main_frame)
        
        # Image display area
        self.create_image_panel(main_frame)
    
    def create_tools_panel(self, parent):
        """Create the tools panel on the left side."""
        left_panel = ttk.LabelFrame(parent, text="Tools")
        left_panel.pack(side=LEFT, fill=Y, padx=5, pady=5)
        
        # Basic operations
        ttk.Button(left_panel, text="Open Image", command=self.controller.open_image).pack(fill=X, pady=5)
        ttk.Button(left_panel, text="Remove Background", command=self.controller.remove_background).pack(fill=X, pady=5)
        
        # Background options
        self.create_background_panel(left_panel)
        
        # Enhancement options
        self.create_enhancement_panel(left_panel)
        
        # Crop options
        self.create_crop_panel(left_panel)
        
        # Save options
        self.create_save_panel(left_panel)
        
        # Reset button
        ttk.Button(left_panel, text="Reset Image", command=self.controller.reset_image).pack(fill=X, pady=10)
    
    def create_background_panel(self, parent):
        """Create the background options panel."""
        bg_frame = ttk.LabelFrame(parent, text="Background Options")
        bg_frame.pack(fill=X, pady=5, padx=5)
        
        ttk.Button(bg_frame, text="Add Solid Background", 
                  command=self.controller.add_solid_background).pack(fill=X, pady=2)
        ttk.Button(bg_frame, text="Choose Color", 
                  command=self.controller.choose_bg_color).pack(fill=X, pady=2)
        
        ttk.Button(bg_frame, text="Add Gradient Background", 
                  command=self.controller.add_gradient_background).pack(fill=X, pady=2)
        ttk.Button(bg_frame, text="Choose Gradient Colors", 
                  command=self.controller.choose_gradient_colors).pack(fill=X, pady=2)
        
        # Gradient direction
        self.controller.gradient_direction = StringVar(value="horizontal")
        direction_frame = ttk.Frame(bg_frame)
        direction_frame.pack(fill=X, pady=2)
        
        ttk.Radiobutton(direction_frame, text="Horizontal", 
                       variable=self.controller.gradient_direction, 
                       value="horizontal").pack(side=LEFT)
        ttk.Radiobutton(direction_frame, text="Vertical", 
                       variable=self.controller.gradient_direction, 
                       value="vertical").pack(side=LEFT)
        ttk.Radiobutton(direction_frame, text="Diagonal", 
                       variable=self.controller.gradient_direction, 
                       value="diagonal").pack(side=LEFT)
    
    def create_enhancement_panel(self, parent):
        """Create the image enhancement panel."""
        enhance_frame = ttk.LabelFrame(parent, text="Image Enhancement")
        enhance_frame.pack(fill=X, pady=5, padx=5)
        
        # Initialize enhancement variables in controller if not already done
        if not hasattr(self.controller, 'brightness_val'):
            self.controller.brightness_val = DoubleVar(value=1.0)
        if not hasattr(self.controller, 'contrast_val'):
            self.controller.contrast_val = DoubleVar(value=1.0)
        if not hasattr(self.controller, 'sharpness_val'):
            self.controller.sharpness_val = DoubleVar(value=1.0)
        if not hasattr(self.controller, 'saturation_val'):
            self.controller.saturation_val = DoubleVar(value=1.0)
        
        # Brightness
        ttk.Label(enhance_frame, text="Brightness:").pack(anchor=W)
        brightness_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=HORIZONTAL,
                                    variable=self.controller.brightness_val, 
                                    command=lambda x: self.controller.update_enhancement())
        brightness_scale.pack(fill=X)
        
        # Contrast
        ttk.Label(enhance_frame, text="Contrast:").pack(anchor=W)
        contrast_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=HORIZONTAL,
                                  variable=self.controller.contrast_val, 
                                  command=lambda x: self.controller.update_enhancement())
        contrast_scale.pack(fill=X)
        
        # Sharpness
        ttk.Label(enhance_frame, text="Sharpness:").pack(anchor=W)
        sharpness_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=HORIZONTAL,
                                   variable=self.controller.sharpness_val, 
                                   command=lambda x: self.controller.update_enhancement())
        sharpness_scale.pack(fill=X)
        
        # Saturation
        ttk.Label(enhance_frame, text="Saturation:").pack(anchor=W)
        saturation_scale = ttk.Scale(enhance_frame, from_=0.1, to=2.0, orient=HORIZONTAL,
                                    variable=self.controller.saturation_val, 
                                    command=lambda x: self.controller.update_enhancement())
        saturation_scale.pack(fill=X)
    
    def create_crop_panel(self, parent):
        """Create the crop options panel."""
        crop_frame = ttk.LabelFrame(parent, text="Crop Image")
        crop_frame.pack(fill=X, pady=5, padx=5)
        
        ttk.Button(crop_frame, text="Start Crop Mode", 
                  command=self.controller.start_crop_mode).pack(fill=X, pady=2)
        ttk.Button(crop_frame, text="Apply Crop", 
                  command=self.controller.apply_crop).pack(fill=X, pady=2)
    
    def create_save_panel(self, parent):
        """Create the save options panel."""
        save_frame = ttk.LabelFrame(parent, text="Save Options")
        save_frame.pack(fill=X, pady=5, padx=5)
        
        self.controller.save_format = StringVar(value="png")
        ttk.Radiobutton(save_frame, text="PNG (with transparency)", 
                       variable=self.controller.save_format, 
                       value="png").pack(anchor=W)
        ttk.Radiobutton(save_frame, text="JPG", 
                       variable=self.controller.save_format, 
                       value="jpg").pack(anchor=W)
        
        ttk.Button(save_frame, text="Save Image", 
                  command=self.controller.save_image).pack(fill=X, pady=5)
    
    def create_image_panel(self, parent):
        """Create the image display panel."""
        self.image_frame = ttk.LabelFrame(parent, text="Image Preview")
        self.image_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = Canvas(self.image_frame, bg="#f0f0f0")
        self.canvas.pack(fill=BOTH, expand=True)
        
        # Scrollbars for canvas
        h_scrollbar = ttk.Scrollbar(self.image_frame, orient=HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=BOTTOM, fill=X)
        
        v_scrollbar = ttk.Scrollbar(self.image_frame, orient=VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Image placeholder
        self.image_label = ttk.Label(self.canvas, text="No image loaded")
        self.canvas.create_window((0, 0), window=self.image_label, anchor=NW)
        
        # Store canvas in controller for access
        self.controller.canvas = self.canvas
    
    def create_status_bar(self):
        """Create the status bar at the bottom of the window."""
        self.controller.status_var = StringVar()
        self.controller.status_var.set("Ready")
        
        self.status_bar = ttk.Label(self.root, textvariable=self.controller.status_var, 
                                   relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)
    
    def setup_bindings(self):
        """Set up event bindings for the application."""
        # Bind canvas events for cropping
        self.canvas.bind("<ButtonPress-1>", self.controller.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.controller.on_crop_motion)
        self.canvas.bind("<ButtonRelease-1>", self.controller.on_crop_release)
        
        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)
    
    def on_window_resize(self, event):
        """Handle window resize events."""
        # Only update if the event is from the root window and not a child widget
        if event.widget == self.root:
            # Add a small delay to avoid too many updates
            self.root.after(100, self.controller.display_image)
    
    def apply_styling(self):
        """Apply styling to the application."""
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")
        style.configure("TLabelframe", padding=5)
        style.configure("TLabelframe.Label", font=('Arial', 10, 'bold'))
        
        # Configure colors
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5")
        style.configure("TButton", background="#e1e1e1")
        
        # Configure hover effect for buttons
        style.map("TButton",
                 foreground=[('pressed', 'blue'), ('active', 'blue')],
                 background=[('pressed', '!disabled', '#d9d9d9'), ('active', '#ececec')])