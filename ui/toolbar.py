import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class Toolbar:
    """
    A customizable toolbar for the image processing application.
    Provides quick access to common functions with icons and tooltips.
    """
    
    def __init__(self, parent, controller):
        """
        Initialize the toolbar.
        
        Args:
            parent: The parent widget
            controller: The controller object that handles the application logic
        """
        self.parent = parent
        self.controller = controller
        
        # Create the toolbar frame
        self.frame = ttk.Frame(parent, relief=tk.RAISED, borderwidth=1)
        self.frame.pack(side=tk.TOP, fill=tk.X)
        
        # Store button references
        self.buttons = {}
        
        # Create toolbar buttons
        self.create_buttons()
        
        # Apply styling
        self.apply_styling()
    
    def create_buttons(self):
        """Create all toolbar buttons with icons and tooltips."""
        # Define button configurations
        button_configs = [
            {
                'name': 'open',
                'icon': 'open.png',
                'tooltip': 'Open Image',
                'command': self.controller.open_image
            },
            {
                'name': 'save',
                'icon': 'save.png',
                'tooltip': 'Save Image',
                'command': self.controller.save_image
            },
            {
                'name': 'separator1',
                'is_separator': True
            },
            {
                'name': 'remove_bg',
                'icon': 'remove_bg.png',
                'tooltip': 'Remove Background',
                'command': self.controller.remove_background
            },
            {
                'name': 'add_bg',
                'icon': 'add_bg.png',
                'tooltip': 'Add Background',
                'command': self.controller.add_solid_background
            },
            {
                'name': 'add_gradient',
                'icon': 'gradient.png',
                'tooltip': 'Add Gradient Background',
                'command': self.controller.add_gradient_background
            },
            {
                'name': 'separator2',
                'is_separator': True
            },
            {
                'name': 'crop',
                'icon': 'crop.png',
                'tooltip': 'Crop Image',
                'command': self.controller.start_crop_mode
            },
            {
                'name': 'enhance',
                'icon': 'enhance.png',
                'tooltip': 'Enhance Image',
                'command': self.show_enhancement_panel
            },
            {
                'name': 'separator3',
                'is_separator': True
            },
            {
                'name': 'reset',
                'icon': 'reset.png',
                'tooltip': 'Reset Image',
                'command': self.controller.reset_image
            },
            {
                'name': 'help',
                'icon': 'help.png',
                'tooltip': 'Help',
                'command': self.controller.show_about
            }
        ]
        
        # Create each button
        for config in button_configs:
            if config.get('is_separator', False):
                self.add_separator()
            else:
                self.add_button(
                    name=config['name'],
                    icon_path=self.get_icon_path(config['icon']),
                    tooltip=config['tooltip'],
                    command=config['command']
                )
    
    def add_button(self, name, icon_path, tooltip, command):
        """
        Add a button to the toolbar.
        
        Args:
            name: Unique name for the button
            icon_path: Path to the button icon
            tooltip: Tooltip text
            command: Function to call when button is clicked
        """
        try:
            # Load icon image
            icon = self.load_icon(icon_path)
            
            # Create button
            button = ttk.Button(self.frame, image=icon, command=command)
            button.pack(side=tk.LEFT, padx=2, pady=2)
            
            # Store icon reference to prevent garbage collection
            button.image = icon
            
            # Add tooltip
            self.create_tooltip(button, tooltip)
            
            # Store button reference
            self.buttons[name] = button
            
        except Exception as e:
            print(f"Error creating button {name}: {str(e)}")
            # Create button without icon as fallback
            button = ttk.Button(self.frame, text=name.capitalize(), command=command)
            button.pack(side=tk.LEFT, padx=2, pady=2)
            self.create_tooltip(button, tooltip)
            self.buttons[name] = button
    
    def add_separator(self):
        """Add a vertical separator to the toolbar."""
        separator = ttk.Separator(self.frame, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
    
    def load_icon(self, icon_path, size=(24, 24)):
        """
        Load an icon image and resize it.
        
        Args:
            icon_path: Path to the icon file
            size: Tuple of (width, height) for the icon
            
        Returns:
            PhotoImage object for the icon
        """
        if os.path.exists(icon_path):
            img = Image.open(icon_path)
            img = img.resize(size, Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        else:
            raise FileNotFoundError(f"Icon file not found: {icon_path}")
    
    def get_icon_path(self, icon_name):
        """
        Get the full path to an icon file.
        
        Args:
            icon_name: Name of the icon file
            
        Returns:
            Full path to the icon file
        """
        # Check if icons directory exists, if not create a default path
        icons_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'icons')
        
        if not os.path.exists(icons_dir):
            # For development, use a fallback path
            icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
            
            # Create directory if it doesn't exist
            if not os.path.exists(icons_dir):
                os.makedirs(icons_dir)
        
        return os.path.join(icons_dir, icon_name)
    
    def create_tooltip(self, widget, text):
        """
        Create a tooltip for a widget.
        
        Args:
            widget: The widget to add a tooltip to
            text: The tooltip text
        """
        tooltip = ToolTip(widget, text)
        
        def enter(event):
            tooltip.show_tip()
        
        def leave(event):
            tooltip.hide_tip()
        
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
    
    def show_enhancement_panel(self):
        """Show or hide the enhancement panel."""
        # This method would toggle the visibility of the enhancement panel
        # Implementation depends on how the enhancement panel is integrated
        if hasattr(self.controller, 'toggle_enhancement_panel'):
            self.controller.toggle_enhancement_panel()
        else:
            # Fallback to just updating enhancements
            self.controller.update_enhancement()
    
    def enable_button(self, name):
        """Enable a toolbar button by name."""
        if name in self.buttons:
            self.buttons[name].configure(state=tk.NORMAL)
    
    def disable_button(self, name):
        """Disable a toolbar button by name."""
        if name in self.buttons:
            self.buttons[name].configure(state=tk.DISABLED)
    
    def apply_styling(self):
        """Apply styling to the toolbar."""
        style = ttk.Style()
        
        # Configure toolbar frame
        style.configure("Toolbar.TFrame", background="#f0f0f0")
        self.frame.configure(style="Toolbar.TFrame")
        
        # Configure toolbar buttons
        style.configure("Toolbar.TButton", 
                       padding=4, 
                       relief="flat",
                       background="#f0f0f0")
        
        # Apply style to all buttons
        for button in self.buttons.values():
            button.configure(style="Toolbar.TButton")


class ToolTip:
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
    
    def show_tip(self):
        """Display the tooltip."""
        if self.tip_window or not self.text:
            return
        
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Create a toplevel window
        self.tip_window = tw = tk.Toplevel(self.widget)
        
        # Remove the window decoration
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        # Create the tooltip label
        label = ttk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=4, ipady=2)
    
    def hide_tip(self):
        """Hide the tooltip."""
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


def create_default_icons():
    """
    Create default placeholder icons if they don't exist.
    This is useful for development when actual icons aren't available.
    """
    icons_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'icons')
    
    if not os.path.exists(icons_dir):
        icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
        
        if not os.path.exists(icons_dir):
            os.makedirs(icons_dir)
    
    # List of default icons to create
    default_icons = [
        'open.png', 'save.png', 'remove_bg.png', 'add_bg.png', 
        'gradient.png', 'crop.png', 'enhance.png', 'reset.png', 'help.png'
    ]
    
    # Create a simple colored square for each icon
    colors = {
        'open.png': (0, 128, 255),      # Blue
        'save.png': (0, 192, 0),        # Green
        'remove_bg.png': (255, 0, 0),   # Red
        'add_bg.png': (128, 0, 128),    # Purple
        'gradient.png': (255, 128, 0),  # Orange
        'crop.png': (0, 128, 128),      # Teal
        'enhance.png': (255, 255, 0),   # Yellow
        'reset.png': (192, 0, 0),       # Dark Red
        'help.png': (0, 64, 128)        # Dark Blue
    }
    
    for icon_name in default_icons:
        icon_path = os.path.join(icons_dir, icon_name)
        
        if not os.path.exists(icon_path):
            # Create a simple colored square as placeholder
            color = colors.get(icon_name, (200, 200, 200))  # Default to gray
            img = Image.new('RGB', (24, 24), color)
            img.save(icon_path)