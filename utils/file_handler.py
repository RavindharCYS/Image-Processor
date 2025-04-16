import os
import io
import json
import time
import shutil
import tempfile
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from PIL import Image, ExifTags
import tkinter as tk
from tkinter import filedialog, messagebox

# Try to import additional libraries for enhanced functionality
try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

try:
    import pyheif
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


class FileHandler:
    """
    A class for handling file operations in the image processing application.
    Provides methods for loading, saving, and managing image files and project files.
    """
    
    def __init__(self, controller=None):
        """
        Initialize the file handler.
        
        Args:
            controller: The controller object that handles the application logic
        """
        self.controller = controller
        
        # Default settings
        self.settings = {
            'recent_files': [],
            'max_recent_files': 10,
            'default_save_format': 'png',
            'default_save_quality': 95,
            'preserve_metadata': True,
            'auto_save_enabled': False,
            'auto_save_interval': 5,  # minutes
            'temp_dir': None,
            'last_directory': None
        }
        
        # Supported file formats
        self.supported_formats = {
            'open': [
                ('All supported formats', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp'),
                ('JPEG files', '*.jpg *.jpeg'),
                ('PNG files', '*.png'),
                ('BMP files', '*.bmp'),
                ('GIF files', '*.gif'),
                ('TIFF files', '*.tiff *.tif'),
                ('WebP files', '*.webp'),
                ('All files', '*.*')
            ],
            'save': [
                ('PNG files', '*.png'),
                ('JPEG files', '*.jpg *.jpeg'),
                ('BMP files', '*.bmp'),
                ('GIF files', '*.gif'),
                ('TIFF files', '*.tiff *.tif'),
                ('WebP files', '*.webp')
            ]
        }
        
        # Add HEIF/HEIC support if available
        if HEIF_AVAILABLE:
            self.supported_formats['open'].insert(1, ('HEIF/HEIC files', '*.heif *.heic'))
        
        # Auto-save timer
        self.auto_save_timer = None
        
        # Create temp directory if needed
        if self.settings['temp_dir'] is None:
            self.settings['temp_dir'] = tempfile.mkdtemp(prefix='image_processor_')
        
        # Current project state
        self.current_file_path = None
        self.project_modified = False
        self.project_data = {
            'version': '1.0',
            'history': [],
            'settings': {}
        }
    
    def open_image(self, file_path=None):
        """
        Open an image file.
        
        Args:
            file_path: Path to the image file or None to show file dialog
            
        Returns:
            PIL Image object or None if failed
        """
        if file_path is None:
            # Show file dialog
            initial_dir = self.settings['last_directory'] or os.path.expanduser('~')
            file_path = filedialog.askopenfilename(
                title="Open Image",
                initialdir=initial_dir,
                filetypes=self.supported_formats['open']
            )
            
            if not file_path:
                return None  # User cancelled
        
        try:
            # Update last directory
            self.settings['last_directory'] = os.path.dirname(file_path)
            
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Handle HEIF/HEIC files if supported
            if file_ext in ['.heif', '.heic'] and HEIF_AVAILABLE:
                heif_file = pyheif.read(file_path)
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
            else:
                # Open with PIL
                image = Image.open(file_path)
            
            # Convert to RGBA for consistency
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Extract metadata if needed
            metadata = None
            if self.settings['preserve_metadata']:
                metadata = self.extract_metadata(file_path, image)
            
            # Add to recent files
            self.add_recent_file(file_path)
            
            # Update current file path
            self.current_file_path = file_path
            self.project_modified = False
            
            # Reset project data
            self.project_data = {
                'version': '1.0',
                'file_path': file_path,
                'original_format': image.format,
                'metadata': metadata,
                'history': [],
                'settings': {}
            }
            
            return image
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error opening file: {str(e)}")
            else:
                messagebox.showerror("Error", f"Could not open file: {str(e)}")
            return None
    
    def save_image(self, image, file_path=None, format=None, quality=None):
        """
        Save an image to a file.
        
        Args:
            image: PIL Image object to save
            file_path: Path to save to or None to show file dialog
            format: File format (e.g., 'PNG', 'JPEG') or None to determine from extension
            quality: Quality for lossy formats (0-100) or None for default
            
        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            # Show file dialog
            initial_dir = self.settings['last_directory'] or os.path.expanduser('~')
            default_ext = f".{self.settings['default_save_format'].lower()}"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Image",
                initialdir=initial_dir,
                defaultextension=default_ext,
                filetypes=self.supported_formats['save']
            )
            
            if not file_path:
                return False  # User cancelled
        
        try:
            # Update last directory
            self.settings['last_directory'] = os.path.dirname(file_path)
            
            # Determine format from extension if not specified
            if format is None:
                format = os.path.splitext(file_path)[1][1:].upper()
                if format == 'JPG':
                    format = 'JPEG'
            
            # Use default quality if not specified
            if quality is None:
                quality = self.settings['default_save_quality']
            
            # Prepare image for saving
            save_image = image.copy()
            
            # Convert to appropriate mode for the format
            if format == 'JPEG':
                if save_image.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', save_image.size, (255, 255, 255))
                    background.paste(save_image, mask=save_image.split()[3])
                    save_image = background
                elif save_image.mode != 'RGB':
                    save_image = save_image.convert('RGB')
            
            # Save with appropriate options
            save_args = {}
            
            if format == 'JPEG':
                save_args['quality'] = quality
                save_args['optimize'] = True
                
                # Add EXIF data if available and preservation is enabled
                if self.settings['preserve_metadata'] and PIEXIF_AVAILABLE:
                    if 'metadata' in self.project_data and self.project_data['metadata']:
                        exif_dict = self.project_data['metadata'].get('exif')
                        if exif_dict:
                            try:
                                exif_bytes = piexif.dump(exif_dict)
                                save_args['exif'] = exif_bytes
                            except Exception as e:
                                print(f"Warning: Could not add EXIF data: {str(e)}")
            
            elif format == 'PNG':
                save_args['optimize'] = True
                
            elif format == 'WEBP':
                save_args['quality'] = quality
                save_args['lossless'] = quality >= 95
                
            elif format == 'TIFF':
                save_args['compression'] = 'tiff_deflate'
            
            # Save the image
            save_image.save(file_path, format=format, **save_args)
            
            # Update current file path
            self.current_file_path = file_path
            self.project_modified = False
            
            # Add to recent files
            self.add_recent_file(file_path)
            
            return True
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error saving file: {str(e)}")
            else:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")
            return False
    
    def export_image(self, image, file_path=None, format=None, quality=None, resize=None):
        """
        Export an image with additional options.
        
        Args:
            image: PIL Image object to export
            file_path: Path to save to or None to show file dialog
            format: File format (e.g., 'PNG', 'JPEG') or None to determine from extension
            quality: Quality for lossy formats (0-100) or None for default
            resize: Tuple of (width, height) to resize to or None for no resize
            
        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            # Show file dialog
            initial_dir = self.settings['last_directory'] or os.path.expanduser('~')
            default_ext = f".{self.settings['default_save_format'].lower()}"
            
            file_path = filedialog.asksaveasfilename(
                title="Export Image",
                initialdir=initial_dir,
                defaultextension=default_ext,
                filetypes=self.supported_formats['save']
            )
            
            if not file_path:
                return False  # User cancelled
        
        try:
            # Update last directory
            self.settings['last_directory'] = os.path.dirname(file_path)
            
            # Determine format from extension if not specified
            if format is None:
                format = os.path.splitext(file_path)[1][1:].upper()
                if format == 'JPG':
                    format = 'JPEG'
            
            # Use default quality if not specified
            if quality is None:
                quality = self.settings['default_save_quality']
            
            # Prepare image for export
            export_image = image.copy()
            
            # Resize if specified
            if resize:
                width, height = resize
                export_image = export_image.resize((width, height), Image.LANCZOS)
            
            # Convert to appropriate mode for the format
            if format == 'JPEG':
                if export_image.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', export_image.size, (255, 255, 255))
                    background.paste(export_image, mask=export_image.split()[3])
                    export_image = background
                elif export_image.mode != 'RGB':
                    export_image = export_image.convert('RGB')
            
            # Save with appropriate options
            save_args = {}
            
            if format == 'JPEG':
                save_args['quality'] = quality
                save_args['optimize'] = True
            elif format == 'PNG':
                save_args['optimize'] = True
            elif format == 'WEBP':
                save_args['quality'] = quality
                save_args['lossless'] = quality >= 95
            elif format == 'TIFF':
                save_args['compression'] = 'tiff_deflate'
            
            # Save the image
            export_image.save(file_path, format=format, **save_args)
            
            # Add to recent files
            self.add_recent_file(file_path)
            
            return True
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error exporting file: {str(e)}")
            else:
                messagebox.showerror("Error", f"Could not export file: {str(e)}")
            return False
    
    def save_project(self, file_path=None):
        """
        Save the current project state to a file.
        
        Args:
            file_path: Path to save to or None to show file dialog
            
        Returns:
            True if successful, False otherwise
        """
        if file_path is None:
            # Show file dialog
            initial_dir = self.settings['last_directory'] or os.path.expanduser('~')
            
            file_path = filedialog.asksaveasfilename(
                title="Save Project",
                initialdir=initial_dir,
                defaultextension=".imprj",
                filetypes=[("Image Processor Project", "*.imprj"), ("All files", "*.*")]
            )
            
            if not file_path:
                return False  # User cancelled
        
        try:
            # Update last directory
            self.settings['last_directory'] = os.path.dirname(file_path)
            
            # Create a temporary directory for project files
            temp_dir = tempfile.mkdtemp(prefix='imprj_')
            
            # Save project data
            project_file = os.path.join(temp_dir, 'project.json')
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(self.project_data, f, indent=2)
            
            # Save current image if available
            if hasattr(self.controller, 'current_image') and self.controller.current_image:
                image_file = os.path.join(temp_dir, 'current.png')
                self.controller.current_image.save(image_file, 'PNG')
            
            # Save original image if available
            if hasattr(self.controller, 'original_image') and self.controller.original_image:
                orig_file = os.path.join(temp_dir, 'original.png')
                self.controller.original_image.save(orig_file, 'PNG')
            
            # Save history images if available
            if hasattr(self.controller, 'history') and self.controller.history:
                history_dir = os.path.join(temp_dir, 'history')
                os.makedirs(history_dir, exist_ok=True)
                
                for i, hist_image in enumerate(self.controller.history):
                    hist_file = os.path.join(history_dir, f'step_{i}.png')
                    hist_image.save(hist_file, 'PNG')
            
            # Create zip file
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path_full = os.path.join(root, file)
                        arcname = os.path.relpath(file_path_full, temp_dir)
                        zipf.write(file_path_full, arcname)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Update current project path
            self.current_file_path = file_path
            self.project_modified = False
            
            # Add to recent files
            self.add_recent_file(file_path)
            
            return True
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error saving project: {str(e)}")
            else:
                messagebox.showerror("Error", f"Could not save project: {str(e)}")
            return False
    
    def open_project(self, file_path=None):
        """
        Open a project file.
        
        Args:
            file_path: Path to the project file or None to show file dialog
            
        Returns:
            Dictionary with project data or None if failed
        """
        if file_path is None:
            # Show file dialog
            initial_dir = self.settings['last_directory'] or os.path.expanduser('~')
            
            file_path = filedialog.askopenfilename(
                title="Open Project",
                initialdir=initial_dir,
                filetypes=[("Image Processor Project", "*.imprj"), ("All files", "*.*")]
            )
            
            if not file_path:
                return None  # User cancelled
        
        try:
            # Update last directory
            self.settings['last_directory'] = os.path.dirname(file_path)
            
            # Create a temporary directory for extracted files
            temp_dir = tempfile.mkdtemp(prefix='imprj_')
            
            # Extract zip file
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Load project data
            project_file = os.path.join(temp_dir, 'project.json')
            with open(project_file, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
            
            # Load current image if available
            current_image = None
            image_file = os.path.join(temp_dir, 'current.png')
            if os.path.exists(image_file):
                current_image = Image.open(image_file).convert('RGBA')
            
            # Load original image if available
            original_image = None
            orig_file = os.path.join(temp_dir, 'original.png')
            if os.path.exists(orig_file):
                original_image = Image.open(orig_file).convert('RGBA')
            
            # Load history images if available
            history = []
            history_dir = os.path.join(temp_dir, 'history')
            if os.path.exists(history_dir):
                i = 0
                while True:
                    hist_file = os.path.join(history_dir, f'step_{i}.png')
                    if os.path.exists(hist_file):
                        hist_image = Image.open(hist_file).convert('RGBA')
                        history.append(hist_image)
                        i += 1
                    else:
                        break
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Update current project path
            self.current_file_path = file_path
            self.project_modified = False
            self.project_data = project_data
            
            # Add to recent files
            self.add_recent_file(file_path)
            
            # Return project data and images
            return {
                'project_data': project_data,
                'current_image': current_image,
                'original_image': original_image,
                'history': history
            }
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error opening project: {str(e)}")
            else:
                messagebox.showerror("Error", f"Could not open project: {str(e)}")
            return None
    
    def extract_metadata(self, file_path, image):
        """
        Extract metadata from an image file.
        
        Args:
            file_path: Path to the image file
            image: PIL Image object
            
        Returns:
            Dictionary with metadata or None if failed
        """
        metadata = {}
        
        try:
            # Basic image info
            metadata['format'] = image.format
            metadata['mode'] = image.mode
            metadata['size'] = image.size
            
            # File info
            file_stat = os.stat(file_path)
            metadata['file_size'] = file_stat.st_size
            metadata['created'] = datetime.fromtimestamp(file_stat.st_ctime).isoformat()
            metadata['modified'] = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            # EXIF data
            if PIEXIF_AVAILABLE:
                try:
                    exif_dict = piexif.load(file_path)
                    if exif_dict:
                        # Convert byte values to strings for JSON serialization
                        for ifd in exif_dict:
                            if isinstance(exif_dict[ifd], dict):
                                for key in list(exif_dict[ifd].keys()):
                                    if isinstance(exif_dict[ifd][key], bytes):
                                        try:
                                            exif_dict[ifd][key] = exif_dict[ifd][key].decode('utf-8')
                                        except UnicodeDecodeError:
                                            # If can't decode, convert to hex string
                                            exif_dict[ifd][key] = exif_dict[ifd][key].hex()
                        
                        metadata['exif'] = exif_dict
                except Exception as e:
                    print(f"Warning: Could not extract EXIF data: {str(e)}")
            
            # PIL's getexif as fallback
            elif hasattr(image, '_getexif'):
                try:
                    exif = image._getexif()
                    if exif:
                        exif_data = {}
                        for tag_id, value in exif.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8')
                                except UnicodeDecodeError:
                                    value = value.hex()
                            exif_data[tag] = value
                        metadata['exif_pil'] = exif_data
                except Exception as e:
                    print(f"Warning: Could not extract PIL EXIF data: {str(e)}")
            
            return metadata
            
        except Exception as e:
            print(f"Warning: Could not extract metadata: {str(e)}")
            return None
    
    def add_recent_file(self, file_path):
        """
        Add a file to the recent files list.
        
        Args:
            file_path: Path to the file
        """
        # Convert to absolute path
        file_path = os.path.abspath(file_path)
        
        # Remove if already in list
        if file_path in self.settings['recent_files']:
            self.settings['recent_files'].remove(file_path)
        
        # Add to beginning of list
        self.settings['recent_files'].insert(0, file_path)
        
        # Trim list if needed
        if len(self.settings['recent_files']) > self.settings['max_recent_files']:
            self.settings['recent_files'] = self.settings['recent_files'][:self.settings['max_recent_files']]
        
        # Save settings
        self.save_settings()
    
    def get_recent_files(self):
        """
        Get the list of recent files.
        
        Returns:
            List of file paths
        """
        # Filter out files that no longer exist
        valid_files = []
        for file_path in self.settings['recent_files']:
            if os.path.exists(file_path):
                valid_files.append(file_path)
        
        # Update the list
        self.settings['recent_files'] = valid_files
        
        return valid_files
    
    def clear_recent_files(self):
        """Clear the recent files list."""
        self.settings['recent_files'] = []
        self.save_settings()
    
    def start_auto_save(self):
        """Start the auto-save timer."""
        if self.settings['auto_save_enabled']:
            interval_ms = self.settings['auto_save_interval'] * 60 * 1000  # Convert minutes to milliseconds
            
            # Cancel existing timer if any
            self.stop_auto_save()
            
            # Start new timer
            self.auto_save_timer = threading.Timer(interval_ms / 1000, self.auto_save)
            self.auto_save_timer.daemon = True
            self.auto_save_timer.start()
    
    def stop_auto_save(self):
        """Stop the auto-save timer."""
        if self.auto_save_timer:
            self.auto_save_timer.cancel()
            self.auto_save_timer = None
    
    def auto_save(self):
        """Perform auto-save."""
        if not self.settings['auto_save_enabled'] or not self.project_modified:
            return
        
        # Check if we have a current file path
        if self.current_file_path:
            # For image files, save a backup
            if os.path.splitext(self.current_file_path)[1].lower() not in ['.imprj']:
                backup_dir = os.path.join(self.settings['temp_dir'], 'auto_save')
                os.makedirs(backup_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.basename(self.current_file_path)
                name, ext = os.path.splitext(filename)
                backup_path = os.path.join(backup_dir, f"{name}_autosave_{timestamp}{ext}")
                
                # Save current image if available
                if hasattr(self.controller, 'current_image') and self.controller.current_image:
                    self.save_image(self.controller.current_image, backup_path)
            
            # For project files, save directly
            else:
                self.save_project(self.current_file_path)
        
        # Restart timer
        self.start_auto_save()
    
    def load_settings(self):
        """
        Load settings from file.
        
        Returns:
            Dictionary with settings
        """
        settings_file = self.get_settings_file_path()
        
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                
                # Update settings with loaded values
                self.settings.update(loaded_settings)
            except Exception as e:
                print(f"Warning: Could not load settings: {str(e)}")
        
        return self.settings
    
    def save_settings(self):
        """
        Save settings to file.
        
        Returns:
            True if successful, False otherwise
        """
        settings_file = self.get_settings_file_path()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Warning: Could not save settings: {str(e)}")
            return False
    
    def get_settings_file_path(self):
        """
        Get the path to the settings file.
        
        Returns:
            Path to settings file
        """
        # Use platform-specific location for settings
        if os.name == 'nt':  # Windows
            app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
            return os.path.join(app_data, 'ImageProcessor', 'settings.json')
        else:  # macOS, Linux
            config_dir = os.path.expanduser('~/.config')
            return os.path.join(config_dir, 'ImageProcessor', 'settings.json')
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.settings['temp_dir'] and os.path.exists(self.settings['temp_dir']):
            try:
                # Remove files older than 7 days
                cutoff_time = time.time() - (7 * 24 * 60 * 60)
                
                for root, dirs, files in os.walk(self.settings['temp_dir']):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) < cutoff_time:
                            os.remove(file_path)
                    
                    # Remove empty directories
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
            
            except Exception as e:
                print(f"Warning: Could not clean up temp files: {str(e)}")
    
    def show_file_info(self, file_path=None):
        """
        Show information about a file.
        
        Args:
            file_path: Path to the file or None to use current file
            
        Returns:
            Dictionary with file information
        """
        if file_path is None:
            file_path = self.current_file_path
        
        if not file_path or not os.path.exists(file_path):
            return None
        
        try:
            file_info = {}
            
            # Basic file info
            file_stat = os.stat(file_path)
            file_info['path'] = file_path
            file_info['name'] = os.path.basename(file_path)
            file_info['directory'] = os.path.dirname(file_path)
            file_info['size'] = file_stat.st_size
            file_info['created'] = datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            file_info['modified'] = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            file_info['extension'] = os.path.splitext(file_path)[1].lower()
            
            # For image files, get image info
            if file_info['extension'] in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp']:
                try:
                    with Image.open(file_path) as img:
                        file_info['image_format'] = img.format
                        file_info['image_mode'] = img.mode
                        file_info['image_size'] = img.size
                        
                        # Get EXIF data if available
                        if hasattr(img, '_getexif'):
                            exif = img._getexif()
                            if exif:
                                exif_data = {}
                                for tag_id, value in exif.items():
                                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                                    if isinstance(value, bytes):
                                        try:
                                            value = value.decode('utf-8')
                                        except UnicodeDecodeError:
                                            value = value.hex()
                                    exif_data[tag] = value
                                file_info['exif'] = exif_data
                except Exception as e:
                    file_info['image_error'] = str(e)
            
            return file_info
            
        except Exception as e:
            if self.controller and hasattr(self.controller, 'show_error'):
                self.controller.show_error(f"Error getting file info: {str(e)}")
            else:
                print(f"Error getting file info: {str(e)}")
            return None
    
    def create_file_info_dialog(self, parent, file_path=None):
        """
        Create a dialog showing file information.
        
        Args:
            parent: Parent window
            file_path: Path to the file or None to use current file
        """
        if file_path is None:
            file_path = self.current_file_path
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showinfo("File Info", "No file selected")
            return
        
        # Get file info
        file_info = self.show_file_info(file_path)
        if not file_info:
            return
        
        # Create dialog
        dialog = tk.Toplevel(parent)
        dialog.title("File Information")
        dialog.geometry("500x400")
        dialog.minsize(400, 300)
        
        # Make it modal
        dialog.transient(parent)
        dialog.grab_set()
        
        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for different categories
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Basic info tab
        basic_tab = ttk.Frame(notebook, padding=10)
        notebook.add(basic_tab, text="Basic Info")
        
        # File info
        row = 0
        ttk.Label(basic_tab, text="File Name:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(basic_tab, text=file_info['name']).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        row += 1
        ttk.Label(basic_tab, text="Location:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(basic_tab, text=file_info['directory']).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        row += 1
        ttk.Label(basic_tab, text="Size:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        size_str = self.format_file_size(file_info['size'])
        ttk.Label(basic_tab, text=size_str).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        row += 1
        ttk.Label(basic_tab, text="Created:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(basic_tab, text=file_info['created']).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        row += 1
        ttk.Label(basic_tab, text="Modified:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(basic_tab, text=file_info['modified']).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        row += 1
        ttk.Label(basic_tab, text="Type:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        file_type = file_info['extension'].upper()[1:] + " File"
        ttk.Label(basic_tab, text=file_type).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Image info tab (if applicable)
        if 'image_format' in file_info:
            image_tab = ttk.Frame(notebook, padding=10)
            notebook.add(image_tab, text="Image Info")
            
            row = 0
            ttk.Label(image_tab, text="Format:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(image_tab, text=file_info['image_format']).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            
            row += 1
            ttk.Label(image_tab, text="Mode:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(image_tab, text=file_info['image_mode']).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            
            row += 1
            ttk.Label(image_tab, text="Dimensions:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            dimensions = f"{file_info['image_size'][0]} × {file_info['image_size'][1]} pixels"
            ttk.Label(image_tab, text=dimensions).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            
            # Try to load and display a thumbnail
            try:
                with Image.open(file_path) as img:
                    # Create a thumbnail
                    thumb_size = (150, 150)
                    img.thumbnail(thumb_size)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    
                    row += 1
                    ttk.Label(image_tab, text="Preview:").grid(row=row, column=0, sticky=tk.NW, padx=5, pady=5)
                    
                    preview_label = ttk.Label(image_tab, image=photo)
                    preview_label.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
                    preview_label.image = photo  # Keep a reference
            except Exception as e:
                row += 1
                ttk.Label(image_tab, text="Preview:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                ttk.Label(image_tab, text="Could not load preview").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        # EXIF data tab (if available)
        if 'exif' in file_info:
            exif_tab = ttk.Frame(notebook, padding=10)
            notebook.add(exif_tab, text="EXIF Data")
            
            # Create a scrollable frame for EXIF data
            exif_canvas = tk.Canvas(exif_tab)
            exif_scrollbar = ttk.Scrollbar(exif_tab, orient=tk.VERTICAL, command=exif_canvas.yview)
            exif_scrollable_frame = ttk.Frame(exif_canvas)
            
            exif_scrollable_frame.bind(
                "<Configure>",
                lambda e: exif_canvas.configure(scrollregion=exif_canvas.bbox("all"))
            )
            
            exif_canvas.create_window((0, 0), window=exif_scrollable_frame, anchor=tk.NW)
            exif_canvas.configure(yscrollcommand=exif_scrollbar.set)
            
            exif_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            exif_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add EXIF data
            row = 0
            for tag, value in file_info['exif'].items():
                # Skip binary data or very long values
                if isinstance(value, (bytes, bytearray)) or (isinstance(value, str) and len(value) > 100):
                    value = "[Binary data]"
                
                ttk.Label(exif_scrollable_frame, text=str(tag) + ":").grid(
                    row=row, column=0, sticky=tk.W, padx=5, pady=2)
                
                # Wrap long text
                if isinstance(value, str) and len(value) > 50:
                    text_widget = tk.Text(exif_scrollable_frame, wrap=tk.WORD, width=40, height=3)
                    text_widget.insert(tk.END, str(value))
                    text_widget.config(state=tk.DISABLED)
                    text_widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                else:
                    ttk.Label(exif_scrollable_frame, text=str(value)).grid(
                        row=row, column=1, sticky=tk.W, padx=5, pady=2)
                
                row += 1
        
        # Close button
        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(pady=10)
        
        # Center the dialog on the parent window
        dialog.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def format_file_size(self, size_bytes):
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def create_batch_processor(self, parent):
        """
        Create a batch processing dialog.
        
        Args:
            parent: Parent window
            
        Returns:
            Batch processor dialog
        """
        from tkinter import filedialog, messagebox
        
        # Create dialog
        dialog = tk.Toplevel(parent)
        dialog.title("Batch Processing")
        dialog.geometry("700x500")
        dialog.minsize(600, 400)
        
        # Make it modal
        dialog.transient(parent)
        dialog.grab_set()
        
        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for different steps
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Files selection tab
        files_tab = ttk.Frame(notebook, padding=10)
        notebook.add(files_tab, text="1. Select Files")
        
        # Source files frame
        source_frame = ttk.LabelFrame(files_tab, text="Source Files")
        source_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Source directory
        source_dir_frame = ttk.Frame(source_frame)
        source_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(source_dir_frame, text="Source Directory:").pack(side=tk.LEFT)
        
        source_dir_var = tk.StringVar()
        source_dir_entry = ttk.Entry(source_dir_frame, textvariable=source_dir_var, width=40)
        source_dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_source_dir():
            directory = filedialog.askdirectory(
                title="Select Source Directory",
                initialdir=self.settings['last_directory'] or os.path.expanduser('~')
            )
            if directory:
                source_dir_var.set(directory)
                self.settings['last_directory'] = directory
                update_file_list()
        
        ttk.Button(source_dir_frame, text="Browse...", command=browse_source_dir).pack(side=tk.LEFT)
        
        # File filter
        filter_frame = ttk.Frame(source_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filter_frame, text="File Types:").pack(side=tk.LEFT)
        
        file_filter_var = tk.StringVar(value="*.jpg;*.jpeg;*.png;*.bmp")
        file_filter_entry = ttk.Entry(filter_frame, textvariable=file_filter_var, width=30)
        file_filter_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(filter_frame, text="Refresh", command=lambda: update_file_list()).pack(side=tk.LEFT)
        
        # Include subdirectories
        include_subdirs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(filter_frame, text="Include Subdirectories", 
                       variable=include_subdirs_var).pack(side=tk.LEFT, padx=10)
        
        # File list
        file_list_frame = ttk.Frame(source_frame)
        file_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(file_list_frame, text="Files to Process:").pack(anchor=tk.W)
        
        file_listbox = tk.Listbox(file_list_frame, selectmode=tk.EXTENDED)
        file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        file_scrollbar = ttk.Scrollbar(file_list_frame, orient=tk.VERTICAL, command=file_listbox.yview)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        file_listbox.config(yscrollcommand=file_scrollbar.set)
        
        # File count label
        file_count_var = tk.StringVar(value="0 files selected")
        ttk.Label(source_frame, textvariable=file_count_var).pack(anchor=tk.W)
        
        # Function to update file list
        def update_file_list():
            file_listbox.delete(0, tk.END)
            
            source_dir = source_dir_var.get()
            if not source_dir or not os.path.isdir(source_dir):
                file_count_var.set("Invalid directory")
                return
            
            # Parse file filters
            filters = file_filter_var.get().split(';')
            filters = [f.strip() for f in filters if f.strip()]
            
            # Find files
            files = []
            
            if include_subdirs_var.get():
                # Walk through subdirectories
                for root, _, filenames in os.walk(source_dir):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        # Check if file matches any filter
                        if any(fnmatch.fnmatch(filename.lower(), filter.lower()) for filter in filters):
                            files.append(file_path)
            else:
                # Only look in the specified directory
                for filter in filters:
                    # Convert wildcard pattern to regex
                    pattern = filter.replace('.', '\\.').replace('*', '.*').replace('?', '.')
                    regex = re.compile(pattern, re.IGNORECASE)
                    
                    # Find matching files
                    for filename in os.listdir(source_dir):
                        if regex.match(filename):
                            files.append(os.path.join(source_dir, filename))
            
            # Add files to listbox
            for file in sorted(files):
                file_listbox.insert(tk.END, file)
            
            # Update count
            file_count = file_listbox.size()
            file_count_var.set(f"{file_count} files selected")
        
        # Operations tab
        operations_tab = ttk.Frame(notebook, padding=10)
        notebook.add(operations_tab, text="2. Choose Operations")
        
        # Operations frame
        operations_frame = ttk.LabelFrame(operations_tab, text="Operations to Apply")
        operations_frame.pack(fill=tk.BOTH, expand=True)
        
        # Resize operation
        resize_var = tk.BooleanVar(value=False)
        resize_frame = ttk.Frame(operations_frame)
        resize_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(resize_frame, text="Resize Images", 
                       variable=resize_var).pack(side=tk.LEFT)
        
        ttk.Label(resize_frame, text="Width:").pack(side=tk.LEFT, padx=(10, 5))
        resize_width_var = tk.IntVar(value=800)
        ttk.Spinbox(resize_frame, from_=1, to=10000, width=5, 
                   textvariable=resize_width_var).pack(side=tk.LEFT)
        
        ttk.Label(resize_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 5))
        resize_height_var = tk.IntVar(value=600)
        ttk.Spinbox(resize_frame, from_=1, to=10000, width=5, 
                   textvariable=resize_height_var).pack(side=tk.LEFT)
        
        # Maintain aspect ratio
        maintain_aspect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(resize_frame, text="Maintain Aspect Ratio", 
                       variable=maintain_aspect_var).pack(side=tk.LEFT, padx=10)
        
        # Format conversion
        convert_var = tk.BooleanVar(value=False)
        convert_frame = ttk.Frame(operations_frame)
        convert_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(convert_frame, text="Convert Format", 
                       variable=convert_var).pack(side=tk.LEFT)
        
        ttk.Label(convert_frame, text="Format:").pack(side=tk.LEFT, padx=(10, 5))
        format_var = tk.StringVar(value="PNG")
        format_combo = ttk.Combobox(convert_frame, textvariable=format_var, width=10)
        format_combo['values'] = ['PNG', 'JPEG', 'BMP', 'TIFF', 'WebP']
        format_combo.pack(side=tk.LEFT)
        
        ttk.Label(convert_frame, text="Quality:").pack(side=tk.LEFT, padx=(10, 5))
        quality_var = tk.IntVar(value=90)
        ttk.Spinbox(convert_frame, from_=1, to=100, width=5, 
                   textvariable=quality_var).pack(side=tk.LEFT)
        
        # Background removal
        bg_remove_var = tk.BooleanVar(value=False)
        bg_frame = ttk.Frame(operations_frame)
        bg_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(bg_frame, text="Remove Background", 
                       variable=bg_remove_var).pack(side=tk.LEFT)
        
        # Check if rembg is available
        try:
            import rembg
            rembg_available = True
        except ImportError:
            rembg_available = False
        
        if not rembg_available:
            ttk.Label(bg_frame, text="(requires rembg package)", 
                     foreground="red").pack(side=tk.LEFT, padx=5)
        
        # Add background color
        bg_color_var = tk.BooleanVar(value=False)
        bg_color_frame = ttk.Frame(operations_frame)
        bg_color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(bg_color_frame, text="Add Background Color", 
                       variable=bg_color_var).pack(side=tk.LEFT)
        
        bg_color_value = tk.StringVar(value="#ffffff")
        bg_color_preview = tk.Canvas(bg_color_frame, width=20, height=20, bg=bg_color_value.get())
        bg_color_preview.pack(side=tk.LEFT, padx=5)
        
        def choose_bg_color():
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Choose Background Color", 
                                         initialcolor=bg_color_value.get())
            if color[1]:
                bg_color_value.set(color[1])
                bg_color_preview.config(bg=color[1])
        
        ttk.Button(bg_color_frame, text="Choose Color", 
                  command=choose_bg_color).pack(side=tk.LEFT)
        
        # Watermark
        watermark_var = tk.BooleanVar(value=False)
        watermark_frame = ttk.Frame(operations_frame)
        watermark_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(watermark_frame, text="Add Watermark Text", 
                       variable=watermark_var).pack(side=tk.LEFT)
        
        ttk.Label(watermark_frame, text="Text:").pack(side=tk.LEFT, padx=(10, 5))
        watermark_text_var = tk.StringVar(value="© Copyright")
        ttk.Entry(watermark_frame, textvariable=watermark_text_var, width=20).pack(side=tk.LEFT)
        
        ttk.Label(watermark_frame, text="Position:").pack(side=tk.LEFT, padx=(10, 5))
        position_var = tk.StringVar(value="bottom-right")
        position_combo = ttk.Combobox(watermark_frame, textvariable=position_var, width=12)
        position_combo['values'] = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center']
        position_combo.pack(side=tk.LEFT)
        
        # Output tab
        output_tab = ttk.Frame(notebook, padding=10)
        notebook.add(output_tab, text="3. Set Output")
        
        # Output directory
        output_frame = ttk.LabelFrame(output_tab, text="Output Options")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        output_dir_frame = ttk.Frame(output_frame)
        output_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_dir_frame, text="Output Directory:").pack(side=tk.LEFT)
        
        output_dir_var = tk.StringVar()
        output_dir_entry = ttk.Entry(output_dir_frame, textvariable=output_dir_var, width=40)
        output_dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_output_dir():
            directory = filedialog.askdirectory(
                title="Select Output Directory",
                initialdir=self.settings['last_directory'] or os.path.expanduser('~')
            )
            if directory:
                output_dir_var.set(directory)
                self.settings['last_directory'] = directory
        
        ttk.Button(output_dir_frame, text="Browse...", command=browse_output_dir).pack(side=tk.LEFT)
        
        # File naming
        naming_frame = ttk.Frame(output_frame)
        naming_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(naming_frame, text="File Naming:").pack(side=tk.LEFT)
        
        naming_var = tk.StringVar(value="original")
        naming_combo = ttk.Combobox(naming_frame, textvariable=naming_var, width=15)
        naming_combo['values'] = ['original', 'original_processed', 'sequence', 'custom']
        naming_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(naming_frame, text="Custom Pattern:").pack(side=tk.LEFT, padx=(10, 5))
        pattern_var = tk.StringVar(value="{name}_{date}")
        pattern_entry = ttk.Entry(naming_frame, textvariable=pattern_var, width=20)
        pattern_entry.pack(side=tk.LEFT)
        
        # Pattern help button
        def show_pattern_help():
            help_text = """
            Available placeholders:
            {name} - Original filename without extension
            {ext} - Original file extension
            {date} - Current date (YYYYMMDD)
            {time} - Current time (HHMMSS)
            {seq} - Sequence number
            {rand} - Random string
            """
            messagebox.showinfo("Pattern Help", help_text)
        
        ttk.Button(naming_frame, text="?", width=2, command=show_pattern_help).pack(side=tk.LEFT, padx=2)
        
        # Overwrite existing files
        overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="Overwrite Existing Files", 
                       variable=overwrite_var).pack(anchor=tk.W, pady=5)
        
        # Create subdirectories based on operations
        create_subdirs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="Create Subdirectories for Each Operation", 
                       variable=create_subdirs_var).pack(anchor=tk.W, pady=5)
        
        # Preview tab
        preview_tab = ttk.Frame(notebook, padding=10)
        notebook.add(preview_tab, text="4. Preview")
        
        # Preview frame
        preview_frame = ttk.LabelFrame(preview_tab, text="Operation Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Preview text
        preview_text = tk.Text(preview_frame, wrap=tk.WORD, width=60, height=15)
        preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=preview_text.yview)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        preview_text.config(yscrollcommand=preview_scrollbar.set)
        
        # Update preview button
        def update_preview():
            preview_text.delete(1.0, tk.END)
            
            # Get selected files
            selected_files = file_listbox.get(0, tk.END)
            if not selected_files:
                preview_text.insert(tk.END, "No files selected.\n")
                return
            
            # Get output directory
            output_dir = output_dir_var.get()
            if not output_dir:
                preview_text.insert(tk.END, "No output directory specified.\n")
                return
            
            # Preview header
            preview_text.insert(tk.END, f"Processing {len(selected_files)} files\n")
            preview_text.insert(tk.END, f"Output directory: {output_dir}\n\n")
            
            # Preview operations
            operations = []
            
            if resize_var.get():
                width = resize_width_var.get()
                height = resize_height_var.get()
                maintain = "maintaining aspect ratio" if maintain_aspect_var.get() else "ignoring aspect ratio"
                operations.append(f"Resize to {width}×{height} pixels ({maintain})")
            
            if convert_var.get():
                format_str = format_var.get()
                quality = quality_var.get()
                operations.append(f"Convert to {format_str} format (quality: {quality}%)")
            
            if bg_remove_var.get():
                operations.append("Remove background")
                
                if not rembg_available:
                    operations[-1] += " (rembg package required)"
            
            if bg_color_var.get():
                color = bg_color_value.get()
                operations.append(f"Add background color: {color}")
            
            if watermark_var.get():
                text = watermark_text_var.get()
                position = position_var.get()
                operations.append(f"Add watermark text: '{text}' at {position}")
            
            # Display operations
            if operations:
                preview_text.insert(tk.END, "Operations to apply:\n")
                for i, op in enumerate(operations, 1):
                    preview_text.insert(tk.END, f"{i}. {op}\n")
            else:
                preview_text.insert(tk.END, "No operations selected.\n")
            
            # File naming preview
            preview_text.insert(tk.END, "\nFile naming preview:\n")
            
            naming_type = naming_var.get()
            if naming_type == "original":
                preview_text.insert(tk.END, "Keep original filenames\n")
            elif naming_type == "original_processed":
                preview_text.insert(tk.END, "Add '_processed' suffix to original filenames\n")
            elif naming_type == "sequence":
                preview_text.insert(tk.END, "Rename files to sequence (img_001, img_002, etc.)\n")
            elif naming_type == "custom":
                pattern = pattern_var.get()
                preview_text.insert(tk.END, f"Custom pattern: {pattern}\n")
                
                # Show example
                import datetime
                sample_name = os.path.basename(selected_files[0])
                name, ext = os.path.splitext(sample_name)
                
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                time_str = datetime.datetime.now().strftime("%H%M%S")
                
                example = pattern
                example = example.replace("{name}", name)
                example = example.replace("{ext}", ext[1:])
                example = example.replace("{date}", date_str)
                example = example.replace("{time}", time_str)
                example = example.replace("{seq}", "001")
                example = example.replace("{rand}", "abc123")
                
                preview_text.insert(tk.END, f"Example: {example}{ext}\n")
            
            # Show a few example files
            preview_text.insert(tk.END, "\nExample files:\n")
            for i, file in enumerate(selected_files[:5]):
                filename = os.path.basename(file)
                preview_text.insert(tk.END, f"{i+1}. {filename}\n")
            
            if len(selected_files) > 5:
                preview_text.insert(tk.END, f"... and {len(selected_files) - 5} more files\n")
        
        ttk.Button(preview_tab, text="Update Preview", command=update_preview).pack(pady=10)
        
        # Process button
        process_button_frame = ttk.Frame(main_frame)
        process_button_frame.pack(fill=tk.X, pady=10)
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(process_button_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Status label
        status_var = tk.StringVar(value="Ready to process")
        status_label = ttk.Label(process_button_frame, textvariable=status_var)
        status_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Process and Cancel buttons
        button_frame = ttk.Frame(process_button_frame)
        button_frame.pack(fill=tk.X)
        
        process_button = ttk.Button(button_frame, text="Process Files", command=lambda: start_processing())
        process_button.pack(side=tk.LEFT)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side=tk.RIGHT)
        
        # Processing variables
        processing_thread = [None]
        cancel_processing = [False]
        
        # Function to start processing
        def start_processing():
            # Validate inputs
            selected_files = file_listbox.get(0, tk.END)
            if not selected_files:
                messagebox.showwarning("Warning", "No files selected.")
                return
            
            output_dir = output_dir_var.get()
            if not output_dir:
                messagebox.showwarning("Warning", "No output directory specified.")
                return
            
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not create output directory: {str(e)}")
                    return
            
            # Check if any operations are selected
            operations_selected = (resize_var.get() or convert_var.get() or bg_remove_var.get() or 
                                  bg_color_var.get() or watermark_var.get())
            
            if not operations_selected:
                messagebox.showwarning("Warning", "No operations selected.")
                return
            
            # Disable UI during processing
            process_button.config(state=tk.DISABLED)
            notebook.config(state=tk.DISABLED)
            
            # Reset progress
            progress_var.set(0)
            status_var.set("Starting processing...")
            cancel_processing[0] = False
            
            # Start processing thread
            processing_thread[0] = threading.Thread(target=process_files)
            processing_thread[0].daemon = True
            processing_thread[0].start()
        
        # Function to process files
        def process_files():
            import time
            import random
            import datetime
            
            try:
                # Get selected files
                selected_files = list(file_listbox.get(0, tk.END))
                total_files = len(selected_files)
                
                # Get output directory
                output_dir = output_dir_var.get()
                
                # Create subdirectories if needed
                if create_subdirs_var.get():
                    if resize_var.get():
                        os.makedirs(os.path.join(output_dir, "resized"), exist_ok=True)
                    if convert_var.get():
                        os.makedirs(os.path.join(output_dir, "converted"), exist_ok=True)
                    if bg_remove_var.get():
                        os.makedirs(os.path.join(output_dir, "bg_removed"), exist_ok=True)
                    if bg_color_var.get():
                        os.makedirs(os.path.join(output_dir, "bg_color"), exist_ok=True)
                    if watermark_var.get():
                        os.makedirs(os.path.join(output_dir, "watermarked"), exist_ok=True)
                
                # Process each file
                for i, file_path in enumerate(selected_files):
                    if cancel_processing[0]:
                        status_var.set("Processing cancelled")
                        break
                    
                    # Update status
                    filename = os.path.basename(file_path)
                    status_var.set(f"Processing {i+1}/{total_files}: {filename}")
                    
                    try:
                        # Open the image
                        img = Image.open(file_path)
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        
                        # Apply operations
                        current_img = img
                        current_dir = output_dir
                        
                        # Resize
                        if resize_var.get():
                            width = resize_width_var.get()
                            height = resize_height_var.get()
                            
                            if maintain_aspect_var.get():
                                # Calculate new dimensions while maintaining aspect ratio
                                img_width, img_height = current_img.size
                                aspect = img_width / img_height
                                
                                if width / height > aspect:
                                    # Height is the limiting factor
                                    new_width = int(height * aspect)
                                    new_height = height
                                else:
                                    # Width is the limiting factor
                                    new_width = width
                                    new_height = int(width / aspect)
                            else:
                                new_width = width
                                new_height = height
                            
                            current_img = current_img.resize((new_width, new_height), Image.LANCZOS)
                            
                            if create_subdirs_var.get():
                                current_dir = os.path.join(output_dir, "resized")
                        
                        # Remove background
                        if bg_remove_var.get():
                            if rembg_available:
                                import rembg
                                
                                # Convert to bytes
                                img_byte_arr = io.BytesIO()
                                current_img.save(img_byte_arr, format='PNG')
                                img_byte_arr.seek(0)
                                
                                # Remove background
                                output = rembg.remove(img_byte_arr.getvalue())
                                
                                # Convert back to PIL Image
                                current_img = Image.open(io.BytesIO(output)).convert("RGBA")
                            
                            if create_subdirs_var.get():
                                current_dir = os.path.join(output_dir, "bg_removed")
                        
                        # Add background color
                        if bg_color_var.get():
                            color = bg_color_value.get()
                            
                            # Create a solid color background
                            bg = Image.new('RGBA', current_img.size, color)
                            
                            # Composite the image with the background
                            current_img = Image.alpha_composite(bg, current_img)
                            
                            if create_subdirs_var.get():
                                current_dir = os.path.join(output_dir, "bg_color")
                        
                        # Add watermark
                        if watermark_var.get():
                            from PIL import ImageDraw, ImageFont
                            
                            text = watermark_text_var.get()
                            position = position_var.get()
                            
                            # Create a copy for drawing
                            watermarked = current_img.copy()
                            draw = ImageDraw.Draw(watermarked)
                            
                            # Try to load a font, fall back to default if not available
                            try:
                                font = ImageFont.truetype("arial.ttf", 20)
                            except IOError:
                                font = ImageFont.load_default()
                            
                            # Calculate text size
                            text_width, text_height = draw.textsize(text, font=font)
                            
                            # Calculate position
                            img_width, img_height = watermarked.size
                            
                            if position == 'top-left':
                                x, y = 10, 10
                            elif position == 'top-right':
                                x, y = img_width - text_width - 10, 10
                            elif position == 'bottom-left':
                                x, y = 10, img_height - text_height - 10
                            elif position == 'bottom-right':
                                x, y = img_width - text_width - 10, img_height - text_height - 10
                            elif position == 'center':
                                x, y = (img_width - text_width) // 2, (img_height - text_height) // 2
                            else:
                                x, y = 10, 10
                            
                            # Add a semi-transparent background for the text
                            padding = 5
                            draw.rectangle(
                                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                                fill=(0, 0, 0, 128)
                            )
                            
                            # Draw the text
                            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
                            
                            current_img = watermarked
                            
                            if create_subdirs_var.get():
                                current_dir = os.path.join(output_dir, "watermarked")
                        
                        # Generate output filename
                        name, ext = os.path.splitext(filename)
                        
                        # Use specified format if converting
                        if convert_var.get():
                            format_str = format_var.get()
                            ext = f".{format_str.lower()}"
                            
                            if create_subdirs_var.get():
                                current_dir = os.path.join(output_dir, "converted")
                        
                        # Generate filename based on naming option
                        naming_type = naming_var.get()
                        
                        if naming_type == "original":
                            output_filename = name + ext
                        elif naming_type == "original_processed":
                            output_filename = f"{name}_processed{ext}"
                        elif naming_type == "sequence":
                            output_filename = f"img_{i+1:03d}{ext}"
                        elif naming_type == "custom":
                            pattern = pattern_var.get()
                            
                            # Replace placeholders
                            date_str = datetime.datetime.now().strftime("%Y%m%d")
                            time_str = datetime.datetime.now().strftime("%H%M%S")
                            rand_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
                            
                            output_name = pattern
                            output_name = output_name.replace("{name}", name)
                            output_name = output_name.replace("{ext}", ext[1:])
                            output_name = output_name.replace("{date}", date_str)
                            output_name = output_name.replace("{time}", time_str)
                            output_name = output_name.replace("{seq}", f"{i+1:03d}")
                            output_name = output_name.replace("{rand}", rand_str)
                            
                            output_filename = f"{output_name}{ext}"
                        else:
                            output_filename = name + ext
                        
                        # Full output path
                        output_path = os.path.join(current_dir, output_filename)
                        
                        # Check if file exists
                        if os.path.exists(output_path) and not overwrite_var.get():
                            # Add a suffix to avoid overwriting
                            base, ext = os.path.splitext(output_path)
                            counter = 1
                            while os.path.exists(f"{base}_{counter}{ext}"):
                                counter += 1
                            output_path = f"{base}_{counter}{ext}"
                        
                        # Save the image
                        save_format = os.path.splitext(output_path)[1][1:].upper()
                        if save_format == 'JPG':
                            save_format = 'JPEG'
                        
                        # Prepare image for saving
                        save_img = current_img
                        
                        # Convert to appropriate mode for the format
                        if save_format == 'JPEG':
                            if save_img.mode == 'RGBA':
                                # Create a white background
                                background = Image.new('RGB', save_img.size, (255, 255, 255))
                                background.paste(save_img, mask=save_img.split()[3])
                                save_img = background
                            elif save_img.mode != 'RGB':
                                save_img = save_img.convert('RGB')
                        
                        # Save with appropriate options
                        save_args = {}
                        
                        if save_format == 'JPEG':
                            save_args['quality'] = quality_var.get()
                            save_args['optimize'] = True
                        elif save_format == 'PNG':
                            save_args['optimize'] = True
                        elif save_format == 'WEBP':
                            save_args['quality'] = quality_var.get()
                            save_args['lossless'] = quality_var.get() >= 95
                        elif save_format == 'TIFF':
                            save_args['compression'] = 'tiff_deflate'
                        
                        save_img.save(output_path, format=save_format, **save_args)
                        
                    except Exception as e:
                        # Log error and continue with next file
                        print(f"Error processing {filename}: {str(e)}")
                        status_var.set(f"Error processing {filename}: {str(e)}")
                        time.sleep(2)  # Show error briefly
                    
                    # Update progress
                    progress = (i + 1) / total_files * 100
                    progress_var.set(progress)
                
                # Processing complete
                if not cancel_processing[0]:
                    status_var.set(f"Processing complete. {total_files} files processed.")
                
                # Re-enable UI
                process_button.config(state=tk.NORMAL)
                notebook.config(state=tk.NORMAL)
                
            except Exception as e:
                # Handle unexpected errors
                status_var.set(f"Error: {str(e)}")
                process_button.config(state=tk.NORMAL)
                notebook.config(state=tk.NORMAL)
        
        # Function to cancel processing
        def cancel_processing_func():
            cancel_processing[0] = True
            status_var.set("Cancelling...")
            
            # Wait for thread to finish
            if processing_thread[0] and processing_thread[0].is_alive():
                processing_thread[0].join(timeout=1.0)
            
            # Close dialog
            dialog.destroy()
        
        # Update cancel button command
        cancel_button.config(command=cancel_processing_func)
        
        # Handle dialog close
        dialog.protocol("WM_DELETE_WINDOW", cancel_processing_func)
        
        # Initialize file list if source directory is already set
        if source_dir_var.get():
            update_file_list()
        
        return dialog
    
    def create_export_dialog(self, parent, image):
        """
        Create a dialog for exporting an image with various options.
        
        Args:
            parent: Parent window
            image: PIL Image to export
            
        Returns:
            Export dialog
        """
        # Create dialog
        dialog = tk.Toplevel(parent)
        dialog.title("Export Image")
        dialog.geometry("500x600")
        dialog.minsize(500, 600)
        
        # Make it modal
        dialog.transient(parent)
        dialog.grab_set()
        
        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image info
        info_frame = ttk.LabelFrame(main_frame, text="Image Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current dimensions
        img_width, img_height = image.size
        ttk.Label(info_frame, text=f"Current Dimensions: {img_width} × {img_height} pixels").pack(anchor=tk.W, pady=2)
        
        # File size (approximate)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        file_size = len(img_byte_arr.getvalue())
        ttk.Label(info_frame, text=f"Approximate File Size: {self.format_file_size(file_size)}").pack(anchor=tk.W, pady=2)
        
        # Export options
        options_frame = ttk.LabelFrame(main_frame, text="Export Options")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Format selection
        format_frame = ttk.Frame(options_frame)
        format_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        
        format_var = tk.StringVar(value="PNG")
        format_combo = ttk.Combobox(format_frame, textvariable=format_var, width=10)
        format_combo['values'] = ['PNG', 'JPEG', 'BMP', 'TIFF', 'WebP']
        format_combo.pack(side=tk.LEFT, padx=5)
        
        # Quality (for lossy formats)
        quality_frame = ttk.Frame(options_frame)
        quality_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(quality_frame, text="Quality:").pack(side=tk.LEFT)
        
        quality_var = tk.IntVar(value=90)
        quality_scale = ttk.Scale(quality_frame, from_=1, to=100, variable=quality_var, orient=tk.HORIZONTAL)
        quality_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        quality_label = ttk.Label(quality_frame, text="90%", width=5)
        quality_label.pack(side=tk.LEFT)
        
        # Update quality label
        def update_quality_label(*args):
            quality_label.config(text=f"{quality_var.get()}%")
        
        quality_var.trace_add("write", update_quality_label)
        
        # Show/hide quality based on format
        def update_quality_visibility(*args):
            format_str = format_var.get()
            if format_str in ['JPEG', 'WebP']:
                quality_frame.pack(fill=tk.X, pady=5)
            else:
                quality_frame.pack_forget()
        
        format_var.trace_add("write", update_quality_visibility)
        update_quality_visibility()  # Initial visibility
        
        # Resize options
        resize_frame = ttk.LabelFrame(options_frame, text="Resize")
        resize_frame.pack(fill=tk.X, pady=5)
        
        resize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(resize_frame, text="Resize Image", variable=resize_var).pack(anchor=tk.W, pady=2)
        
        dimensions_frame = ttk.Frame(resize_frame)
        dimensions_frame.pack(fill=tk.X, pady=2)
        
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
        ttk.Checkbutton(resize_frame, text="Maintain Aspect Ratio", variable=aspect_var).pack(anchor=tk.W, pady=2)
        
        # Common sizes
        common_frame = ttk.Frame(resize_frame)
        common_frame.pack(fill=tk.X, pady=2)
        
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
            if aspect_var.get():
                new_width = width_var.get()
                new_height = int(new_width / original_aspect)
                height_var.set(new_height)
        
        def update_width(*args):
            if aspect_var.get():
                new_height = height_var.get()
                new_width = int(new_height * original_aspect)
                width_var.set(new_width)
        
        width_var.trace_add("write", update_height)
        height_var.trace_add("write", update_width)
        
        # Metadata options
        metadata_frame = ttk.LabelFrame(options_frame, text="Metadata")
        metadata_frame.pack(fill=tk.X, pady=5)
        
        preserve_var = tk.BooleanVar(value=self.settings['preserve_metadata'])
        ttk.Checkbutton(metadata_frame, text="Preserve Metadata (when possible)", 
                       variable=preserve_var).pack(anchor=tk.W, pady=2)
        
        # Output location
        output_frame = ttk.LabelFrame(main_frame, text="Output Location")
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        output_dir_frame = ttk.Frame(output_frame)
        output_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_dir_frame, text="Save To:").pack(side=tk.LEFT)
        
        output_dir_var = tk.StringVar(value=self.settings['last_directory'] or os.path.expanduser('~'))
        output_dir_entry = ttk.Entry(output_dir_frame, textvariable=output_dir_var, width=40)
        output_dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_output_dir():
            directory = filedialog.askdirectory(
                title="Select Output Directory",
                initialdir=output_dir_var.get()
            )
            if directory:
                output_dir_var.set(directory)
        
        ttk.Button(output_dir_frame, text="Browse...", command=browse_output_dir).pack(side=tk.LEFT)
        
        # Filename
        filename_frame = ttk.Frame(output_frame)
        filename_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filename_frame, text="Filename:").pack(side=tk.LEFT)
        
        # Generate default filename
        default_filename = "image_export"
        if self.current_file_path:
            base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
            default_filename = f"{base_name}_export"
        
        filename_var = tk.StringVar(value=default_filename)
        filename_entry = ttk.Entry(filename_frame, textvariable=filename_var, width=30)
        filename_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # File extension label (updates based on format)
        ext_label = ttk.Label(filename_frame, text=".png")
        ext_label.pack(side=tk.LEFT)
        
        # Update extension label when format changes
        def update_ext_label(*args):
            format_str = format_var.get().lower()
            ext_label.config(text=f".{format_str}")
        
        format_var.trace_add("write", update_ext_label)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create a canvas for preview
        preview_canvas = tk.Canvas(preview_frame, bg="#f0f0f0")
        preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview image reference
        preview_image_ref = [None]
        
        # Function to update preview
        def update_preview():
            # Create a copy of the image
            preview_img = image.copy()
            
            # Apply resize if enabled
            if resize_var.get():
                new_width = width_var.get()
                new_height = height_var.get()
                preview_img = preview_img.resize((new_width, new_height), Image.LANCZOS)
            
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
            preview_img = preview_img.resize((preview_width, preview_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(preview_img)
            
            # Clear canvas and display image
            preview_canvas.delete("all")
            preview_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            
            # Store reference to prevent garbage collection
            preview_image_ref[0] = photo
            
            # Update file size estimate
            format_str = format_var.get()
            quality = quality_var.get()
            
            # Estimate file size
            img_byte_arr = io.BytesIO()
            save_img = preview_img
            
            # Convert to appropriate mode for the format
            if format_str == 'JPEG':
                if save_img.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', save_img.size, (255, 255, 255))
                    background.paste(save_img, mask=save_img.split()[3])
                    save_img = background
                elif save_img.mode != 'RGB':
                    save_img = save_img.convert('RGB')
            
            # Save with appropriate options
            save_args = {}
            
            if format_str == 'JPEG':
                save_args['quality'] = quality
                save_args['optimize'] = True
            elif format_str == 'PNG':
                save_args['optimize'] = True
            elif format_str == 'WEBP':
                save_args['quality'] = quality
                save_args['lossless'] = quality >= 95
            elif format_str == 'TIFF':
                save_args['compression'] = 'tiff_deflate'
            
            save_img.save(img_byte_arr, format=format_str, **save_args)
            estimated_size = len(img_byte_arr.getvalue())
            
            # Update info label
            info_label.config(text=f"Dimensions: {img_width} × {img_height} pixels | "
                                  f"Estimated Size: {self.format_file_size(estimated_size)}")
        
        # Info label for dimensions and file size
        info_label = ttk.Label(preview_frame, text="")
        info_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Update preview when canvas is configured
        def on_canvas_configure(event):
            update_preview()
        
        preview_canvas.bind("<Configure>", on_canvas_configure)
        
        # Update preview when options change
        for var in [format_var, quality_var, resize_var, width_var, height_var, aspect_var]:
            var.trace_add("write", lambda *args: update_preview())
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Export button
        def on_export():
            try:
                # Get output path
                output_dir = output_dir_var.get()
                filename = filename_var.get()
                format_str = format_var.get()
                ext = f".{format_str.lower()}"
                
                # Full output path
                output_path = os.path.join(output_dir, filename + ext)
                
                # Check if file exists
                if os.path.exists(output_path):
                    overwrite = messagebox.askyesno(
                        "File Exists",
                        f"The file {filename + ext} already exists. Do you want to overwrite it?"
                    )
                    if not overwrite:
                        return
                
                # Prepare image for export
                export_img = image.copy()
                
                # Apply resize if enabled
                if resize_var.get():
                    new_width = width_var.get()
                    new_height = height_var.get()
                    export_img = export_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Export the image
                quality = quality_var.get()
                preserve_metadata = preserve_var.get()
                
                # Update settings
                self.settings['preserve_metadata'] = preserve_metadata
                self.settings['last_directory'] = output_dir
                self.save_settings()
                
                # Export with appropriate options
                result = self.export_image(
                    export_img, 
                    output_path, 
                    format=format_str, 
                    quality=quality
                )
                
                if result:
                    messagebox.showinfo("Export Complete", f"Image exported successfully to:\n{output_path}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Export Failed", "Failed to export image. Please check the settings and try again.")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"An error occurred during export:\n{str(e)}")
        
        ttk.Button(button_frame, text="Export", command=on_export).pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Initialize preview
        dialog.update_idletasks()
        update_preview()
        
        # Center the dialog on the parent window
        dialog.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        return dialog


def test_file_handler():
    """Test function for the file handler."""
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Create a simple UI for testing
        root = tk.Tk()
        root.title("File Handler Test")
        root.geometry("800x600")
        
        # Create the file handler
        handler = FileHandler()
        
        # Load settings
        handler.load_settings()
        
        # Create a frame for controls
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Current image
        current_image = [None]
        
        # Open image function
        def open_image():
            img = handler.open_image()
            if img:
                current_image[0] = img
                update_preview(img)
                status_var.set(f"Opened: {os.path.basename(handler.current_file_path)}")
        
        ttk.Button(control_frame, text="Open Image", command=open_image).pack(pady=5)
        
        # Save image function
        def save_image():
            if current_image[0]:
                result = handler.save_image(current_image[0])
                if result:
                    status_var.set(f"Saved: {os.path.basename(handler.current_file_path)}")
        
        ttk.Button(control_frame, text="Save Image", command=save_image).pack(pady=5)
        
        # Export image function
        def export_image():
            if current_image[0]:
                dialog = handler.create_export_dialog(root, current_image[0])
                root.wait_window(dialog)
        
        ttk.Button(control_frame, text="Export Image", command=export_image).pack(pady=5)
        
        # File info function
        def show_file_info():
            if handler.current_file_path:
                handler.create_file_info_dialog(root)
            else:
                status_var.set("No file loaded")
        
        ttk.Button(control_frame, text="File Info", command=show_file_info).pack(pady=5)
        
        # Batch processing function
        def show_batch_dialog():
            dialog = handler.create_batch_processor(root)
            root.wait_window(dialog)
        
        ttk.Button(control_frame, text="Batch Processing", command=show_batch_dialog).pack(pady=5)
        
        # Recent files function
        def show_recent_files():
            recent_files = handler.get_recent_files()
            if recent_files:
                menu = tk.Menu(root, tearoff=0)
                for file in recent_files:
                    menu.add_command(
                        label=os.path.basename(file),
                        command=lambda f=file: open_recent_file(f)
                    )
                menu.add_separator()
                menu.add_command(label="Clear Recent Files", command=handler.clear_recent_files)
                
                # Show the menu
                menu.post(recent_btn.winfo_rootx(), recent_btn.winfo_rooty() + recent_btn.winfo_height())
            else:
                status_var.set("No recent files")
        
        def open_recent_file(file_path):
            img = handler.open_image(file_path)
            if img:
                current_image[0] = img
                update_preview(img)
                status_var.set(f"Opened: {os.path.basename(file_path)}")
        
        recent_btn = ttk.Button(control_frame, text="Recent Files", command=show_recent_files)
        recent_btn.pack(pady=5)
        
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
        
        # Clean up on exit
        def on_exit():
            handler.cleanup_temp_files()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_exit)
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_file_handler()